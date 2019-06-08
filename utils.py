from .data import image_name
import numpy as np
from string import ascii_uppercase
from prettytable import PrettyTable

def explore_subject_names(people_names):
    h = dict()
    for c in ascii_uppercase:
        h[c] = sum([p.startswith(c) for p in people_names])
        # print(h)
    return h

def print_dataset_stat(paths_labels, ds_name='path label', label_name=['Different', 'Same']):
    print(f'Examples for {ds_name} dataset structure:')
    print('############')
    print('Raw view:')
    print(paths_labels[:3])
    print('############')
    print('Pretty view:')
    print([tuple((image_name(p[0]), image_name(p[1]), label_name[p[2]])) for p in paths_labels[:3]])
    ds_size = len(paths_labels)
    print('############')
    print(f'Dataset size: {ds_size}')
    print(f'{len(label_name)} classes:')
    labels = np.array([ l for _, _, l in paths_labels])
    t = PrettyTable(['Class', 'Size', 'Percentage'])
    for i,name in enumerate(label_name):
        t.add_row([name, f'{len(labels[labels==i])}/{ds_size}', f'{len(labels[labels==i])/ds_size*100:.2f}%'])
    print(t)