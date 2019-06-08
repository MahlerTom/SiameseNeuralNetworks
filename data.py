from .preprocess import load_and_preprocess_from_path_label, load_and_preprocess_image

import os
import random
import IPython.display as display
import tensorflow as tf

def load_data(data_path, labels_file, print_imgs=3):
  import pathlib
  
  # Get the full path of the directory
  data_root = pathlib.Path(data_path)

  # Create list of all paths in directory  
  data_paths = list(data_root.glob('*'))
  data_paths = [ str(path) for path in data_paths ] # Trasnform paths to strings
  data_count = len(data_paths)
  
  # Print relevant information  
  print('Loaded', data_count, 'image paths')
  if print_imgs > 0:
    print('##########################################')
    print('Printing Example Images')
    print()
    
    for n in range(print_imgs):
      image_path = random.choice(data_paths)
      display.display(display.Image(image_path))
      print(image_path.split('/')[-1][:-4])
      print()
      
    print('##########################################')
  
  # Insert the right tuple based on the structure (left, right, label)
  rel_path = os.path.join('/', *data_paths[0].split('/')[:-1])
  data_paths_paths = list()
  with open(labels_file) as lf:
    for l in lf:
      l = l[:-1].split('\t')
      if len(l) >= 3:
        left = os.path.join(rel_path, l[0] + '_' + l[1].zfill(4) + '.jpg')
        same = (len(l) == 3)
        right_path = l[0] + '_' + l[2].zfill(4) if same else l[2] + '_' + l[3].zfill(4)        
        right = os.path.join(rel_path, right_path + '.jpg')          
        data_paths_paths.append(tuple((left, right, int(same))))

  return data_paths_paths

# Utility function to print the name of the image easily (tested on Google Colab)
def image_name(path):
    return path.split('/')[-1][:-4]

def split_train_val_paths(paths_labels, letters):
    train = list()
    val = list()  
    for p in paths_labels:
        left_name = image_name(p[0])
        train_append = True
        for l in letters:
            if left_name.startswith(l):
                train_append = False
                break;
        (train if train_append else val).append(p)

    return train, val

def create_ds(image_paths, labels, norm=None, _resize=None, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, tf.cast(labels, tf.bool)))
    label_ds = ds.map(
        lambda paths, label: load_and_preprocess_from_path_label(paths, label, norm=norm, _resize=_resize), 
        num_parallel_calls=AUTOTUNE
    )
    return label_ds

def create_image_ds(image_paths, norm=None, _resize=None, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    # print(path_ds)
    image_ds = path_ds.map(lambda path: load_and_preprocess_image(path, norm=norm, _resize=_resize), num_parallel_calls=AUTOTUNE)
    return image_ds

def create_label_ds(labels, dtype=tf.bool):
    return tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.bool))

def prepare_ds(ds, batch_size, buffer_size, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    # Setting a shuffle buffer size as large as the dataset ensures that the data is
    # completely shuffled.
    ds = ds.shuffle(buffer_size=buffer_size)
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size)
    
    # `prefetch` lets the dataset fetch batches, in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def split_train_val(ds_pre, ds_len, val_size=.3):
    val_ds_pre_len = int(val_size * ds_len)
    val_ds_pre = ds_pre.take(val_ds_pre_len)
    train_ds_pre = ds_pre.skip(val_ds_pre_len)
    return train_ds_pre, val_ds_pre

def init_ds(ds, images_labels_paths, norm=255.0, _resize=[256, 256], batch_size=1, buffer_size=None, verbose=0, AUTOTUNE=tf.data.experimental.AUTOTUNE):
    if ds is None:
        if buffer_size is None:
            buffer_size = len(images_labels_paths)
        image_paths = [ (p[0], p[1]) for p in images_labels_paths ]
        labels = [ p[2] for p in images_labels_paths ]
        ds = create_ds(image_paths=image_paths, labels=labels, norm=norm, _resize=_resize, AUTOTUNE=AUTOTUNE)
        ds = prepare_ds(ds, batch_size=batch_size, buffer_size=buffer_size, AUTOTUNE=AUTOTUNE)
            
        if verbose > 0:
            print(ds)
    return ds