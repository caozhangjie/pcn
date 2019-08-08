# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views

from tensorpack import dataflow
from io_util import read_pcd, save_pcd
import numpy as np

import pdb

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def read(args):
    df = dataflow.LMDBSerializer.load(args.lmdb_path, shuffle=False)
    size = df.size()
    df.reset_state()
    data_gen = df.get_data()
    i=0
    for _, data in enumerate(data_gen):
        i+=1
        name, inputs, gt = data
        synset_id = name.split('_')[0]
        os.system('mkdir -p data/shapenet_pcd/train/partial/'+synset_id)
        os.system('mkdir -p data/shapenet_pcd/train/gt/'+synset_id)
        name = name.split('_')[1]
        save_pcd('data/shapenet_pcd/train/partial/'+synset_id+'/'+name+'.pcd', np.array(inputs))
        save_pcd('data/shapenet_pcd/train/gt/'+synset_id+'/'+name+'.pcd', np.array(gt))
        print(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', default='data/shapenet/train.lmdb')
    args = parser.parse_args()

    read(args)
