import os
import random
import IPython.display as display

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