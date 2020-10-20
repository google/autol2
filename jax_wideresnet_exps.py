# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-
from jax.api import grad, jit, vmap, pmap, device_put
from jax.api import pmap
from jax import random, lax
from jax.lib import xla_bridge
from jax.tree_util import tree_map, tree_multimap, tree_reduce, tree_flatten, tree_unflatten
from jax.experimental import stax
from jax.experimental import optimizers
from jax.config import config
import jax.numpy as np
from functools import partial
import time
import jaxwrn_utils as utils
from absl import app, flags, logging
import pandas as pd
import os
import numpy as onp


"Training details"
flags.DEFINE_integer('seed', 1, 'seed')
flags.DEFINE_string('losst','xentr', 'loss type')
flags.DEFINE_float('epochs',200,'epochs')
flags.DEFINE_boolean('physical',False,'If true, epochs correspond to t.eta')
flags.DEFINE_boolean('physicalL2',False,'If true, epochs correspond to t.eta.lambda')
flags.DEFINE_integer('bs', 128, 'bs per device')
flags.DEFINE_float('lr',0.2,'learning rate')
flags.DEFINE_boolean('std_wrn_sch',False,'Learning rate 3 decays')
flags.DEFINE_boolean('momentum',True,'Momentum=0.9')
flags.DEFINE_float('L2',0.0005,'L2')

"Model"
flags.DEFINE_integer('N',4,'depth')
flags.DEFINE_integer('K',10,'width')

"Data Details"
flags.DEFINE_boolean('augment',True,'turn on data augmentation')
flags.DEFINE_boolean('mix',True,'use mixup')

"Measurements and model loading"
flags.DEFINE_boolean('TPU',True,'Use tpu')
flags.DEFINE_string('jobdir',None,'job dir')
flags.DEFINE_integer('meas_step',10,'measure every X steps')
flags.DEFINE_integer('checkpointing',0,'save checkpoints every X steps. if =/=0 will automatically load last checkpoint')
flags.DEFINE_string('load_w',None,'load weights from file')
flags.DEFINE_boolean('steps_from_load',False,'If true count total epochs from the point where the model is loaded')

"AutoL2 details"
flags.DEFINE_boolean('L2_sch',False,'AutoL2 Schedule')
flags.DEFINE_float('L2dec',10.0,'decay factor for L2')
flags.DEFINE_float('delay',0.1,'Length of the refractory period in units of epochs.eta.lambda')

FLAGS = flags.FLAGS

def main(unused_argv):
    from jax.api import grad, jit, vmap, pmap, device_put
    "The following is required to use TPU Driver as JAX's backend."

    if FLAGS.TPU:
      config.FLAGS.jax_xla_backend = "tpu_driver"
      config.FLAGS.jax_backend_target = "grpc://" + os.environ['TPU_ADDR'] + ':8470'
      TPU_ADDR=os.environ['TPU_ADDR']    
    ndevices = xla_bridge.device_count()
    if not FLAGS.TPU:
        ndevices=1

    pmap = partial(pmap, axis_name='i')

    """Setup some experiment parameters."""
    meas_step=FLAGS.meas_step
    training_epochs = int(FLAGS.epochs)

    tmult=1.0
    if FLAGS.physical:
      tmult=FLAGS.lr
      if FLAGS.physicalL2:
        tmult=FLAGS.L2*tmult
    if FLAGS.physical:
      training_epochs=1+int(FLAGS.epochs/tmult)
    
    print('Evolving for {:}e'.format(training_epochs))
    losst=FLAGS.losst
    learning_rate=FLAGS.lr 
    batch_size_per_device=FLAGS.bs
    N=FLAGS.N
    K=FLAGS.K



    batch_size = batch_size_per_device * ndevices
    steps_per_epoch = 50000 // batch_size
    training_steps = training_epochs * steps_per_epoch
    
    
    "Filename from FLAGS"

    filename='wrnL2_'+losst+'_n'+str(N)+'_k'+str(K)
    if FLAGS.momentum:
        filename+='_mom'      
    if FLAGS.L2_sch:
        filename+='_L2sch'+'_decay'+str(FLAGS.L2dec)+'_del'+str(FLAGS.delay)
    if FLAGS.seed!=1:
        filename+='seed'+str(FLAGS.seed)  
    filename+='_L2'+str(FLAGS.L2) 
    if FLAGS.std_wrn_sch:
      filename+='_stddec'
      if FLAGS.physical:
          filename+='phys'   
    else:
      filename+='_ctlr'
    if not FLAGS.augment:
        filename+='_noaug'
    if not FLAGS.mix:
        filename+='_nomixup'
    filename+='_bs'+str(batch_size)+'_lr'+str(learning_rate)
    if FLAGS.jobdir is not None:
        filedir=os.path.join('wrnlogs',FLAGS.jobdir)
    else:
        filedir='wrnlogs'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    filedir=os.path.join(filedir,filename+'.csv')


    print('Saving log to ',filename)
    print('Found {} cores.'.format(ndevices))

    """Load CIFAR10 data and create a minimal pipeline."""

    train_images, train_labels, test_images, test_labels = utils.load_data('cifar10')
    train_images = np.reshape(train_images, (-1, 32, 32 * 3))
    train = (train_images, train_labels)
    test = (test_images, test_labels)
    k=train_labels.shape[-1]
    train = utils.shard_data(train,ndevices)
    test = utils.shard_data(test,ndevices)


    """Create a Wide Resnet and replicate its parameters across the devices."""
    
    initparams, f, _ = utils.WideResnetnt(N, K, k)
    

    "Loss and optimizer definitions"

    l2_norm = lambda params: tree_map(lambda x: np.sum(x ** 2), params)
    l2_reg = lambda params: tree_reduce(lambda x, y: x + y, l2_norm(params))
    currL2 = FLAGS.L2 
    L2p=pmap(lambda x: x)(currL2*np.ones((ndevices,)))

    def xentr(params, images_and_labels):
      images, labels = images_and_labels
      return -np.mean(stax.logsoftmax(f(params, images)) * labels)

    def mse(params, data_tuple):
      """MSE loss."""
      x, y = data_tuple
      return 0.5 * np.mean((y - f(params, x)) ** 2)

    if losst=='xentr':
        print('Using xentr')
        lossm=xentr
    else:
        print('Using mse')
        lossm=mse
    
    loss=lambda params, data, L2: lossm(params,data)+L2*l2_reg(params)

    def accuracy(params, images_and_labels):
      images, labels = images_and_labels 
      return np.mean(
        np.array(
            np.argmax(f(params, images), axis=1) == np.argmax(labels, axis=1), 
            dtype=np.float32)
        )

    "Define optimizer"

    if FLAGS.std_wrn_sch:
        lr=learning_rate
        first_epoch=int(60/200*training_epochs)
        learning_rate_fn=optimizers.piecewise_constant(np.array([1,2,3])*first_epoch*steps_per_epoch,np.array([lr,lr*0.2,lr*0.2**2,lr*0.2**3]))
    else:
      learning_rate_fn=optimizers.make_schedule(learning_rate)

    if FLAGS.momentum:
        momentum=0.9
    else:
        momentum=0



    @pmap
    def update_step(step, state, batch_state,L2):
      batch, batch_state = batch_fn(batch_state)
      params = get_params(state)
      dparams = grad_loss(params, batch, L2)
      dparams = tree_map(lambda x: lax.psum(x, 'i') / ndevices, dparams)
      return step + 1, apply_fn(step, dparams, state), batch_state

    @pmap
    def evaluate(state, data,L2):
     params = get_params(state)
     lossmm=lossm(params,data)
     l2mm=l2_reg(params)
     return lossmm+L2*l2mm, accuracy(params, data),lossmm,l2mm
   
    "Initialization and loading"

    _, params = initparams(random.PRNGKey(0), (-1, 32, 32, 3))
    replicate_array = lambda x: \
        np.broadcast_to(x, (ndevices,) + x.shape)
    replicated_params = tree_map(replicate_array, params)

    grad_loss = jit(grad(loss))
    init_fn, apply_fn, get_params = optimizers.momentum(learning_rate_fn, momentum)
    apply_fn = jit(apply_fn)
    key = random.PRNGKey(FLAGS.seed)
    
    batchinit_fn, batch_fn = utils.sharded_minibatcher(batch_size, ndevices, transform=FLAGS.augment,k=k,mix=FLAGS.mix)
    
    batch_state = pmap(batchinit_fn)(random.split(key, ndevices), train)
    state = pmap(init_fn)(replicated_params)
    
    if FLAGS.checkpointing:
      ## Loading of checkpoint if available/provided.
        single_state=init_fn(params)
        i0, load_state, load_params, filename0, batch_stateb = utils.load_weights(filename,single_state,params,full_file=FLAGS.load_w,ndevices=ndevices)
        if i0 is not None:
            filename=filename0
            if batch_stateb is not None:
              batch_state=batch_stateb           
            if load_params is not None:
              state=pmap(init_fn)(load_params)
            else:
              state=load_state
        else:
            i0=0 
    else:
        i0=0

    if FLAGS.steps_from_load:
        training_steps=i0+training_steps

    
    batch_xs, _ = pmap(batch_fn)(batch_state)
    
    train_loss = []
    train_accuracy = []
    lrL = []
    test_loss = []
    test_accuracy = []
    test_L2,test_lm,train_lm,train_L2=[],[],[],[]
    L2_t=[]
    idel0=i0
    start = time.time()

    step = pmap(lambda x: x)(i0*np.ones((ndevices,)))

    "Start training loop"
    if FLAGS.checkpointing:
      print('Evolving for {:}e and saving every {:}s'.format(training_epochs,FLAGS.checkpointing)) 

    print('Epoch\tLearning Rate\tTrain bareLoss\t L2_norm \tTest Loss\tTrain Error\tTest Error\tTime / Epoch')  
    
    for i in range(i0,training_steps):
        if i % meas_step == 0:
          # Make Measurement
            l, a,lm,L2m = evaluate(state, test,L2p)
            test_loss += [np.mean(l)]
            test_accuracy += [np.mean(a)]
            test_lm += [np.mean(lm)]
            test_L2 += [np.mean(L2m)]
            train_batch, _ = pmap(batch_fn)(batch_state)
            l, a, lm, L2m = evaluate(state, train_batch,L2p)
    
            train_loss += [np.mean(l)]
            train_accuracy += [np.mean(a)]
            train_lm += [np.mean(lm)]
            train_L2 += [np.mean(L2m)]
            L2_t.append(currL2)
            lrL += [learning_rate_fn(i)]
            
            if FLAGS.L2_sch and i>FLAGS.delay/currL2+idel0 and len(train_lm)>2 and ((minloss<=train_lm[-1] and minloss<=train_lm[-2]) or (maxacc>=train_accuracy[-1] and maxacc>=train_accuracy[-2])):
            # If AutoL2 is on and we are beyond the refractory period, decay if the loss or error have increased in the last two measurements.
                print('Decaying L2 to',currL2/FLAGS.L2dec)
                currL2=currL2/FLAGS.L2dec
                L2p=pmap(lambda x: x)(currL2*np.ones((ndevices,)))
                idel0=i            
                    
            elif FLAGS.L2_sch and len(train_lm)>=2:
            # Update the minimum values.
                try:
                  maxacc=max(train_accuracy[-2],maxacc)
                  minloss=min(train_lm[-2],minloss)
                except:
                  maxacc,minloss=train_accuracy[-2],train_lm[-2]
           
      
            if i % (meas_step*10) == 0 or i==i0:
            # Save measurements to csv  
                epoch = batch_size * i / 50000
                dt = (time.time() - start) / (meas_step*10) * steps_per_epoch
                print(('{}\t' + ('{: .4f}\t' * 7)).format(
                    epoch, learning_rate_fn(i), train_lm[-1],train_L2[-1], test_loss[-1], 
                    train_accuracy[-1], test_accuracy[-1], dt))

                start = time.time()
                data={'train_loss':train_loss,'test_loss': test_loss, 'train_acc':train_accuracy,'test_acc':test_accuracy}
                data['train_bareloss']=train_lm
                data['train_L2']=train_L2
                data['test_bareloss']=test_lm
                data['test_L2']=test_L2
                data['L2_t']=L2_t
                df=pd.DataFrame(data)
                
                df['learning_rate']=lrL
                df['width']=K
                df['batch_size']=batch_size
                df['step']=i0+onp.arange(0,len(train_loss))*meas_step

                df.to_csv(filedir, index=False)
          

        if FLAGS.checkpointing:
          ### SAVE MODEL
            if i % FLAGS.checkpointing  ==0 and i>i0:
                
                if not os.path.exists('weights/'):
                  os.makedirs('weights/')
                saveparams=tree_flatten(state[0])[0]
                if ndevices>1:
                    saveparams=[el[0] for el in saveparams]
                saveparams=np.concatenate([el.reshape(-1) for el in saveparams])
                
                step0=i
                print('Step',i)
                print('saving at',filename, step0 ,'size:',saveparams.shape)

                utils.save_weights(filename, step0, saveparams, batch_state)
        


        ## UPDATE
        step, state, batch_state = update_step(step, state, batch_state,L2p)
        
    
    print('Training done')  
    
    if FLAGS.TPU:
      with open('done/'+TPU_ADDR, 'w') as fp:
        fp.write(filedir)  
        pass


if __name__ == '__main__':
  app.run(main)
