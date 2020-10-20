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

from jax.tree_util import tree_flatten, tree_unflatten
from jax import lax
import pickle,os     
import jax.numpy as np
from functools import partial
import numpy as onp
from jax.experimental import stax
from neural_tangents import stax as stax_nt
from collections import namedtuple
from jax.tree_util import register_pytree_node
from jax.api import vmap, pmap
from jax import random
import tensorflow_datasets as tfds


"Saving and loading models"

OptimizerState = namedtuple("OptimizerState", ["packed_state", "tree_def", "subtree_defs"] )
register_pytree_node(OptimizerState, lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)), lambda data, xs: OptimizerState(xs[0], data[0], data[1]))



def flatten_jax(v):
  "Flatten pytree to vector"
  v=tree_flatten(v)[0]
  v=[el.reshape(-1) for el in v]
  return np.concatenate(v)

def unflatten_jax(flat_tensor, orig_tensors):
  "Unflatten vector to pytree"
  orig_tensors,treedef=tree_flatten(orig_tensors)
  unflattened = []
  offset = 0
  for t in orig_tensors:
    num_elems = np.prod(t.shape)
    unflattened.append(np.reshape(flat_tensor[offset:offset + num_elems], t.shape))
    offset += num_elems
  return tree_unflatten(treedef,unflattened)

def save_weights(file_name, step, params_vec,batch_state,saveonlylast=True,savebatch_state=True):
    """Save params and step in appropiate format."""
    filetag='_step'+str(step)

    
    np.save('weights/'+file_name+filetag+'.npy',params_vec)
    if savebatch_state:
        with open('weights/'+file_name+filetag+'.pkl','wb') as f:
            pickle.dump(batch_state,f)

    if not saveonlylast:
        return
    for el in sorted(os.listdir('weights/')):
        if el.startswith(file_name) and ( el!=file_name+filetag+'.npy' and el!=file_name+filetag+'.pkl'):
            ll=[el2 for el2 in os.listdir('weights/') if el2.startswith(file_name)]
            if len(ll)>4:
              os.remove('weights/'+el)
              print('Deleted',el)
    return


def load_weights(filename,discarded_state, discarded_params,full_file=None,ndevices=1):
    """Load time step and weights."""
    from jax.tree_util import tree_map
    if full_file:
        filename0=full_file
        filename+='_prelo_'+full_file[8:-4]
        step=int(full_file.split('_')[-1][4:-4])
        
    else:
      
      if not os.path.exists('weights/'):
        os.makedirs('weights/')
      STEPLIST={int(el2[4:-4]):'_'.join(el.split('_')[:-1]) for el in os.listdir('weights/') if el.startswith(filename) for el2 in el.split('_') if el2.startswith('step') }
      
      if len(STEPLIST.keys())==0:
        return None,None, None, None, None
      while True: 
        step=max(STEPLIST.keys())
        filename=STEPLIST[step]
        print('Loading weight',step, filename)
        filename0='weights/'+filename+'_step'+str(step)
        try:
          flatten_state=np.load(filename0+'.npy')
          break
        except:
          print('Cant load file, going back to previous')
          del STEPLIST[step]  
          pass

    try:      
      with open(filename0+'.pkl','rb') as f:
        batch_state= pickle.load(f)
    except:
      print('Not loading batch state')
      batch_state=None
   
    discarded_state,tree,subtrees=discarded_state

    flatten_state=np.load(filename0+'.npy')

    try:     
      new_state = unflatten_jax(flatten_state, discarded_state)
      new_state=tree_map(lambda x: np.broadcast_to(x,(ndevices,)+x.shape),new_state)
      new_state=OptimizerState(new_state, tree, subtrees)  
      new_params=None
    except:
      print("Old format, loading params not state")
      new_params=unflatten_jax(flatten_state,discarded_params)
      new_params=tree_map(lambda x: np.broadcast_to(x,(ndevices,)+x.shape),new_params)
      new_state=None
    
    
    return step, new_state, new_params, filename+'_s0'+str(step), batch_state
    
" WideResnet Model "

def WideResnetnt(block_size, k, num_classes,batchnorm='std'): #, batch_norm=None,layer_norm=None,freezelast=None):
  """Based off of WideResnet from paper, with or without BatchNorm. 
  (Set config.wrn_block_size=3, config.wrn_widening_f=10 in that case).
  Uses default weight and bias init."""
  parameterization = 'standard' 
  layers_lst = [stax_nt.Conv(16, (3,3), padding='SAME', parameterization=parameterization), 
  WideResnetGroupnt(block_size, 16*k,  parameterization=parameterization,batchnorm=batchnorm),
  WideResnetGroupnt(block_size, 32*k,(2, 2),  parameterization=parameterization,batchnorm=batchnorm),
  WideResnetGroupnt(block_size, 64*k,(2, 2),  parameterization=parameterization,batchnorm=batchnorm)
  ]
  layers_lst += [_batch_norm_internal(batchnorm), stax_nt.Relu()]
  layers_lst += [stax_nt.AvgPool((8,8)), stax_nt.Flatten(), stax_nt.Dense(num_classes, parameterization=parameterization)]
  return stax_nt.serial(*layers_lst)

def WideResnetGroupnt(num_blocks, channels, strides=(1,1), batchnorm=True, parameterization='ntk'):
  """A WideResnet group."""
  blocks = []
  blocks += [WideResnetBlocknt(channels, strides, channel_mismatch=True, batchnorm=batchnorm, parameterization=parameterization)]
  for _ in range(num_blocks-1):
    blocks += [WideResnetBlocknt(channels, (1,1), batchnorm=batchnorm, parameterization=parameterization)]
  return stax_nt.serial(*blocks)

def WideResnetBlocknt(channels, strides=(1,1), channel_mismatch=False, batchnorm='std', parameterization='ntk'):
  """A WideResnet block, with or without BatchNorm."""
  
  Main = stax_nt.serial(_batch_norm_internal(batchnorm), stax_nt.Relu(), stax_nt.Conv(channels, (3,3), strides, padding='SAME', parameterization=parameterization),_batch_norm_internal(batchnorm), stax_nt.Relu(), stax_nt.Conv(channels, (3,3), padding='SAME', parameterization=parameterization))
  
  Shortcut = stax_nt.Identity() if not channel_mismatch else stax_nt.Conv(channels, (3,3), strides, padding='SAME', parameterization=parameterization)
  return stax_nt.serial(stax_nt.FanOut(2), stax_nt.parallel(Main, Shortcut), stax_nt.FanInSum())

def _batch_norm_internal(batchnorm,axis=(0, 1, 2)):
  """Layer constructor for a stax.BatchNorm layer with dummy kernel computation.
  Do not use kernels for architectures that include this function."""
  bn=stax.BatchNorm()
  init_fn, apply_fn = bn
  kernel_fn = lambda kernels: kernels
  return init_fn, apply_fn, kernel_fn



"""Data Pipeline"""

def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return onp.array(x[:, None] == onp.arange(k), dtype)

def _process_shard(images, labels):
  images = onp.reshape(onp.array(images, onp.float32), (-1, 3, 32, 32))
  images = onp.transpose(images, (0, 2, 3, 1))
  labels = onp.array(labels, onp.int32)
  return images, labels

def load_data(dataset='cifar10'):
  ds_train, ds_test = tfds.as_numpy(tfds.load(dataset, data_dir='data/',split=["train", "test"], batch_size=-1, as_dataset_kwargs={"shuffle_files": False}))
  
  train_xs, train_ys, test_xs, test_ys = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])

  train_xs = (train_xs - onp.mean(train_xs)) / onp.std(train_xs)
  test_xs = (test_xs - onp.mean(test_xs)) / onp.std(test_xs)
  print(train_xs.shape,train_ys.shape,max(train_ys))
  k=max(train_ys)+1
  print('There are {:} categories'.format(k))
  train_ys = _one_hot(train_ys, k)
  test_ys = _one_hot(test_ys, k)

  return train_xs, train_ys, test_xs, test_ys

def shard_data(data,ndevices):
  data_per_device, ragged = divmod(data[0].shape[0], ndevices)

  if ragged:
    assert NotImplementedError('Cannot split data evenly across devies.')

  data = list(map(lambda x: np.reshape(x, (ndevices, -1) + x.shape[1:]), data))
  data = list(map(pmap(lambda x: x), data))

  return data

def sharded_minibatcher(batch_size, ndevices, transform,k,mix=True):
  batch_size_per_device, ragged = divmod(batch_size, ndevices)

  if ragged:
    raise NotImplementedError('Cannot divide batch evenly across devices.')

  def shuffle(key_and_data):
    key, data = key_and_data
    key, subkey = random.split(key)
    datapoints_per_device = data[0].shape[0]
    indices = np.arange(datapoints_per_device)
    perm = random.shuffle(subkey, indices)
    return key, list(map(lambda x: x[perm], data)), 0

  def init_fn(key, data):
    datapoints_per_device = data[0].shape[0]

    key, data, i = shuffle((key, data))

    num_batches = datapoints_per_device // batch_size_per_device

    return (key, data, i, num_batches)

  def batch_fn(state):
    key, data, i, num_batches = state

    slice_start = ([i * batch_size_per_device, 0, 0], [i * batch_size_per_device, 0])
    slice_size = ([batch_size_per_device, 32, 32 * 3], [batch_size_per_device, k])
    batch = [
      lax.dynamic_slice(x, start, size) for x, start, size in 
      zip(data, slice_start, slice_size)
    ]

    
    key, subkey = random.split(key)
    batch = augment(subkey, batch, transform,mix)
    

    i = i + 1
    key, data, i = lax.cond(
        i >= num_batches, 
        (key, data), shuffle,
        (key, data, i), lambda x: x)

    return batch, (key, data, i, num_batches,)

  return init_fn, batch_fn

def crop(key, image_and_label):
  """Random flips and crops."""
  image, label = image_and_label

  pixels = 4
  pixpad = (pixels, pixels)
  zero = (0, 0)
  padded_image = np.pad(image, (pixpad, pixpad, zero), 'constant', 0.0)
  corner = random.randint(key, (2,), 0, 2 * pixels)
  corner = np.concatenate((corner, np.zeros((1,), np.int32)))
  img_size = (32, 32, 3)
  cropped_image = lax.dynamic_slice(padded_image, corner, img_size)

  return cropped_image, label
crop = vmap(crop, 0, 0)


def mixup(key, alpha, image_and_label):
  image, label = image_and_label 

  N = image.shape[0]

  weight = random.beta(key, alpha, alpha, (N, 1))
  mixed_label = weight * label + (1.0 - weight) * label[::-1]

  weight = np.reshape(weight, (N, 1, 1, 1))
  mixed_image = weight * image + (1.0 - weight) * image[::-1]

  return mixed_image, mixed_label


def augment(key, image_and_label,transform=True,mix=True):
  image, label = image_and_label
  
  key, split = random.split(key)

  N = image.shape[0]
  image = np.reshape(image, (N, 32, 32, 3))

  if not transform:
      return image, label

  image = np.where(
      random.uniform(split, (N, 1, 1, 1)) < 0.5,
      image[:, :, ::-1],
      image)

  key, split = random.split(key)
  batch_split = random.split(split, N)
  image, label = crop(batch_split, (image, label))
  if mix:
    return mixup(key, 1.0, (image, label))
  else:
    return image, label


