#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:42:00 2018

@author: vnguye04

Calculate mean of AIS dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle
import os
import sys
sys.path.append("./data/")
dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_train.pkl"
import tensorflow as tf


#lat_bins = 300; lon_bins = 300; sog_bins = 30; cog_bins = 72
LAT_BINS = 350; LON_BINS = 1050; SOG_BINS = 30; COG_BINS = 72


def sparse_AIS_to_dense(msgs_,num_timesteps, mmsis):
    def create_dense_vect(msg,lat_bins = 300, lon_bins = 300, sog_bins = 30 ,cog_bins = 72): 
        lat, lon, sog, cog = msg[0], msg[1], msg[2], msg[3]
        data_dim = lat_bins + lon_bins + sog_bins + cog_bins
        dense_vect = np.zeros(data_dim)
        dense_vect[int(lat*lat_bins)] = 1.0
        dense_vect[int(lon*lon_bins) + lat_bins] = 1.0
        dense_vect[int(sog*sog_bins) + lat_bins + lon_bins] = 1.0
        dense_vect[int(cog*cog_bins) + lat_bins + lon_bins + sog_bins] = 1.0       
        return dense_vect
    dense_msgs = []
    for msg in msgs_:
        dense_msgs.append(create_dense_vect(msg,
                                            lat_bins = LAT_BINS,
                                            lon_bins = LON_BINS,
                                            sog_bins = SOG_BINS ,
                                            cog_bins = COG_BINS))
    dense_msgs = np.array(dense_msgs)
    return dense_msgs, num_timesteps, mmsis

dirname = os.path.dirname(dataset_path)

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)

with tf.gfile.Open(dataset_path, "r") as f:
    Vs = pickle.load(f)


data_dim = LAT_BINS + LON_BINS + SOG_BINS + COG_BINS

mean_all = np.zeros((data_dim,))
sum_all = np.zeros((data_dim,))
total_ais_msg = 0

current_mean = np.zeros((0,data_dim))
current_ais_msg = 0

count = 0
for mmsi in Vs.keys():
    count += 1
    print(count)
    tmp = Vs[mmsi][:,[LAT,LON,SOG,COG]]
    tmp[tmp == 1] = 0.99999
    current_sparse_matrix,_,_ = sparse_AIS_to_dense(tmp,0,0)
#    current_mean = np.mean(current_sparse_matrix,axis = 0)
    sum_all += np.sum(current_sparse_matrix,axis = 0)
    total_ais_msg += len(current_sparse_matrix)

mean = sum_all/total_ais_msg

with open(dirname + "/mean.pkl","wb") as f:
    pickle.dump(mean,f)

