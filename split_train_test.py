# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:44:39 2019

@author: Tom
"""

import shutil

train_pairs = []
train_names = set()
with open("trainPairs.txt") as train:
  train_files = train.readlines()[1:]
  
for pair in train_files:
  pair = pair[:-1].split('\t')
  if len(pair) == 3:
    name1 = pair[0]
    id1 = pair[1].zfill(4)
    name2 = name1
    id2 = pair[2].zfill(4)
    
    train_names.add(name1)
    
  elif len(pair) == 4:
    name1 = pair[0]
    id1 = pair[1].zfill(4)
    name2 = pair[2]
    id2 = pair[3].zfill(4)
    
    train_names.add(name1)
    train_names.add(name2)
  train_pairs.append((name1 + '_' + id1, name2 + '_' + id2))
train_names = list(train_names)
#print(train_files)

test_pairs = []
test_names = set()
with open("testPairs.txt") as test:
  test_files = test.readlines()[1:]
  
for pair in test_files:
  pair = pair[:-1].split('\t')
  if len(pair) == 3:
    name1 = pair[0]
    id1 = pair[1].zfill(4)
    name2 = name1
    id2 = pair[2].zfill(4)
    
    test_names.add(name1)
    
  elif len(pair) == 4:
    name1 = pair[0]
    id1 = pair[1].zfill(4)
    name2 = pair[2]
    id2 = pair[3].zfill(4)
    
    test_names.add(name1)

    test_names.add(name2)
  test_pairs.append((name1 + '_' + id1, name2 + '_' + id2))
test_names = list(test_names)

src = "lfw2/"
train_dir = "data/train/"
test_dir = "data/test/"
unused_dir = "data/unused/"

#for name in train_names:
#  print("Moving " + name + "...")
#  shutil.move(src + name, train_dir + name)
#
#for name in test_names:
#  print("Moving " + name + "...")
#  shutil.move(src + name, test_dir + name)

import os

train_dst = 'flatten/train/'
test_dst = 'flatten/test/'
unused_dst = 'flatten/unused/'

def flatten(src, dst):
    for d in os.listdir(src):
        for f in os.listdir(src + d):
            print("Moving " + f + "...")
            shutil.move(src + d + '/' + f, dst + f)

#flatten(train_dir, train_dst)
#flatten(test_dir, test_dst)
flatten(unused_dir, unused_dst)