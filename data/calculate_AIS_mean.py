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
dataset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/mt314/dataset8/dataset8_train.pkl"
import tensorflow as tf
from datasets import sparse_AIS_to_dense

dirname = os.path.dirname(dataset_path)

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)

with tf.gfile.Open(dataset_path, "r") as f:
    Vs = pickle.load(f)

#latlon_bins = 500;sog_bins = 10;cog_bins = 36
#data_dim = 2*latlon_bins + sog_bins + cog_bins
lat_bins = 300; lon_bins = 300; sog_bins = 30; cog_bins = 72
data_dim = lat_bins + lon_bins + sog_bins + cog_bins

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

