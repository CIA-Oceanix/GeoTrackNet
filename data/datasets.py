#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:51:56 2018

@author: vnguye04

Input pipelines script for Tensorflow graph
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('..')
import os
import pickle
import tensorflow as tf


LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)


# The default number of threads used to process data in parallel.
DEFAULT_PARALLELISM = 12

def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis):
    lat_bins = 300; lon_bins = 300; sog_bins = 30; cog_bins = 72
    def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72): 
        lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
        data_dim = lat_bins + lon_bins + sog_bins + cog_bins
        dense_vect = np.zeros(data_dim)
        dense_vect[int(lat*lat_bins)] = 1.0
        dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
        dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
        dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0       
        return dense_vect
#    msgs_[msgs_ == 1] = 0.99999
    dense_msgs = []
    for msg in msgs_:
        dense_msgs.append(create_dense_vect(msg,
                                            lat_bins = lat_bins,
                                            lon_bins = lon_bins,
                                            sog_bins = sog_bins,
                                            cog_bins = cog_bins))
    dense_msgs = np.array(dense_msgs)
    return dense_msgs, num_timesteps, mmsis

def create_AIS_dataset(dataset_path,
                       split,
                       batch_size,
                       data_dim,
                       num_parallel_calls=DEFAULT_PARALLELISM,
                       shuffle=True,
                       repeat=True):
    # Load the data from disk.
    with tf.gfile.Open(dataset_path, "r") as f:
        raw_data = pickle.load(f)
        
    num_examples = len(raw_data)
    dirname = os.path.dirname(dataset_path)
    with open(dirname + "/mean.pkl","r") as f:
        mean = pickle.load(f)

    def aistrack_generator():
        for k in raw_data.keys():
            tmp = raw_data[k][::2,[LAT,LON,SOG,COG]] # 10 min
            tmp[tmp == 1] = 0.99999
            yield tmp, len(tmp), raw_data[k][0,MMSI] 

    dataset = tf.data.Dataset.from_generator(
                                  aistrack_generator,
                                  output_types=(tf.float64, tf.int64, tf.int64))
    
    if repeat: dataset = dataset.repeat()
    if shuffle: dataset = dataset.shuffle(num_examples)
    
    dataset = dataset.map(
            lambda msg_, num_timesteps, mmsis: tuple(tf.py_func(sparse_AIS_to_dense,
                                                   [msg_, num_timesteps, mmsis],
                                                   [tf.float64, tf.int64, tf.int64])),
                                                num_parallel_calls=num_parallel_calls)
    
    # Batch sequences togther, padding them to a common length in time.
    dataset = dataset.padded_batch(batch_size,
                                 padded_shapes=([None, data_dim], [], []))
    
    
    def process_AIS_batch(data, lengths, mmsis):
        """Create mean-centered and time-major next-step prediction Tensors."""
        data = tf.to_float(tf.transpose(data, perm=[1, 0, 2]))
        lengths = tf.to_int32(lengths)
        mmsis = tf.to_int32(mmsis)
        targets = data
        
        # Mean center the inputs.
        inputs = data - tf.constant(mean, dtype=tf.float32,
                                    shape=[1, 1, mean.shape[0]])
        # Shift the inputs one step forward in time. Also remove the last 
        # timestep so that targets and inputs are the same length.
        inputs = tf.pad(data, [[1, 0], [0, 0], [0, 0]], mode="CONSTANT")[:-1] 
        # Mask out unused timesteps.
        inputs *= tf.expand_dims(tf.transpose(
            tf.sequence_mask(lengths, dtype=inputs.dtype)), 2)
        return inputs, targets, lengths, mmsis
    
    dataset = dataset.map(process_AIS_batch,
                          num_parallel_calls=num_parallel_calls)
#    dataset = dataset.prefetch(num_examples)
    dataset = dataset.prefetch(50)
    itr = dataset.make_one_shot_iterator()
    inputs, targets, lengths, mmsis = itr.get_next()
    return inputs, targets, lengths, mmsis, tf.constant(mean, dtype=tf.float32)

