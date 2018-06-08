#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:13:06 2018

@author: vnguye04

Preprocessing script for MultitaskAIS
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
import os
sys.path.append("..")
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO
from pyproj import Geod
geod = Geod(ellps='WGS84')
#import utm

# Golf of Mexico
#LAT_MIN = 26.5
#LAT_MAX = 30.0
#LON_MIN = -97.5
#LON_MAX = -87


## Bretagne
LAT_MIN = 47.0
LAT_MAX = 50.0
LON_MIN = -7.0
LON_MAX = -4.0

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)

# DATA PATH

dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/"
filename_list = [os.path.join(dataset_path,"010203_position.pkl")] 
dict_list = []
for filename in filename_list:
    with open(filename,"rb") as f:
        temp = pickle.load(f)
        dict_list.append(temp)

## Uncomment if you want to create shapefile
#for Vi,zone in zip(dict_list, zone_list):
#    filename = data_path + month + "/Zone" + zone + "_2014_01.shp"
#    print("Creating " + filename + "...")
#    utils.createShapefile(filename,Vi)
    
# STEP1: REMOVING ABNORMAL TIMESTAMPS AND ABNORMAL SPEEDS AND MERGING ZONES
###############################################################################
print("REMOVING ABNORMAL TIMESTAMPS AND ABNORMAL SPEEDS AND MERGING ZONES...")
print("CHANGING BOUNDARY (LAT, LON)...")
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
Vs = dict()
for Vi,filename in zip(dict_list, filename_list):
    print(filename)
    for mmsi in Vi.keys():
        # Boundary
        lat_idx = np.logical_or((Vi[mmsi][:,LAT] > LAT_MAX),
                                (Vi[mmsi][:,LAT] < LAT_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lat_idx)]        
        lon_idx = np.logical_or((Vi[mmsi][:,LON] > LON_MAX),
                                (Vi[mmsi][:,LON] < LON_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lon_idx)]  
        # Abnormal timestamps
        abnormal_timestamp_idx = np.logical_or((Vi[mmsi][:,TIMESTAMP] > t_max),
                                               (Vi[mmsi][:,TIMESTAMP] < t_min))
        Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_timestamp_idx)]
        # Abnormal speeds
        abnormal_speed_idx = Vi[mmsi][:,SOG] > SPEED_MAX
        Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_speed_idx)]
        # Deleting empty keys
        if len(Vi[mmsi]) == 0:
            del Vi[mmsi]
            continue
        if mmsi not in Vs.keys():
            Vs[mmsi] = Vi[mmsi]
            del Vi[mmsi]
        else:
            Vs[mmsi] = np.concatenate((Vs[mmsi],Vi[mmsi]),axis = 0)
            del Vi[mmsi]
#del dict_list, Vi, abnormal_speed_idx, abnormal_timestamp_idx
             

# STEP 2: Cutting discontinuous voyages into smaller voyages 
###############################################################################
print("Cutting discontinuous voyages into smaller voyages...")
count = 0
voyages = dict()
INTERVAL_MAX = 2*3600 # 2h
for mmsi in Vs.keys():
    v = Vs[mmsi]
    # Intervals between successive messages in a track
    intervals = v[1:,TIMESTAMP] - v[:-1,TIMESTAMP]
    idx = np.where(intervals > INTERVAL_MAX)[0] 
    if len(idx) == 0:
        voyages[count] = v
        count += 1
    else:
        tmp = np.split(v,idx+1)
        for t in tmp:
            voyages[count] = t
            count += 1
            

# STEP 3: Removing AIS track whose length is smaller than 20 or who lasts less than 4h
###############################################################################
print("Removing AIS track whose length is smaller than 20 or who lasts less than 4h...")
for mmsi in voyages.keys():
    duration = voyages[mmsi][-1,TIMESTAMP] - voyages[mmsi][0,TIMESTAMP]
    if (len(voyages[mmsi]) < 20) or (duration < 4*3600):
        voyages.pop(mmsi, None)

# STEP 4: Removing anomalous message
###############################################################################
print("Removing anomalous message...")
error_count = 0
tick = time.time()
for mmsi in voyages.keys(): 
    print(mmsi,'...')
    track = voyages[mmsi][:,[TIMESTAMP,LAT,LON,SOG]] # [Timestamp, Lat, Lon, Speed]
    try:
        o_report, o_calcul = utils.detectOutlier(track, speed_max = 30)
        if o_report.all() or o_calcul.all():
            voyages.pop(mmsi, None)
        else:
            voyages[mmsi] = voyages[mmsi][np.invert(o_report)]
            voyages[mmsi] = voyages[mmsi][np.invert(o_calcul)]
    except:
        voyages.pop(mmsi,None)
        error_count += 1
tok = time.time()
print("STEP 4: duration = ",(tok - tick)/60) # 0.33280659914 mins

## STEP 5: Removing 'moored' or 'ar anchor' tracks
###############################################################################
print("Removing 'moored' or 'ar anchor' tracks...")
for mmsi in voyages.keys():
    d_L = float(len(voyages[mmsi]))
    
    if np.count_nonzero(voyages[mmsi][:,NAV_STT] == 1)/d_L > 0.7\
       or np.count_nonzero(voyages[mmsi][:,NAV_STT] == 5)/d_L > 0.7:
        voyages.pop(mmsi,None)
        continue
    sog_max = np.max(voyages[mmsi][:,SOG])
    if sog_max < 1.0:
        voyages.pop(mmsi,None)


# Step 6: Sampling, resolution = 5 min
###############################################################################
tick = time.time()
print('Sampling...')
Vs = dict()
count = 0
for k in voyages.keys():
    v = voyages[k]
    print(count)
    sampling_track = np.empty((0, 9))
    for t in range(int(v[0,TIMESTAMP]), int(v[-1,TIMESTAMP]), 300): # 5 min
        tmp = utils.interpolate(t,v)
        if tmp is not None:
            sampling_track = np.vstack([sampling_track, tmp])
        else:
            sampling_track = None
            break
    if sampling_track is not None:
        Vs[count] = sampling_track
        count += 1
tok = time.time()

print("Removing 'low speed' tracks") 
for mmsi in Vs.keys():
    d_L = float(len(Vs[mmsi]))    
    if np.count_nonzero(Vs[mmsi][:,SOG] < 2)/d_L > 0.8:
        Vs.pop(mmsi,None)

# STEP 7: Re-Splitting
###############################################################################
print('Re-Splitting...')
Data = dict() 
count = 0
for mmsi in Vs.keys(): # 
    v = Vs[mmsi]
    # Split AIS track into small tracks whose duration <= 1 day
    idx = np.arange(0, len(v), 12*24)[1:]
    tmp = np.split(v,idx)
    for subtrack in tmp:
        # only use tracks whose duration >= 4 hours
        if len(subtrack) > 12*4: 
            Data[count] = subtrack
            count += 1

# Step 7: Normalisation 
###############################################################################
print('Normalisation...')
for k in Data.keys():
    v = Data[k]
    v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
    v[:,SOG] = v[:,SOG]/SPEED_MAX
    v[:,COG] = v[:,COG]/360.0

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/mt314/dataset8/dataset8_test.pkl","wb") as f:
    pickle.dump(Data,f)
    
# Step 7bis: Density normalisation
###############################################################################
#with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset7/dataset7_full.pkl","rb") as f:
#    Vs = pickle.load(f)

#Tiles = dict()
#for d_i in range(10):
#    for d_j in range(10):
#        Tiles[str(d_i)+str(d_j)] = []   
#for key in Vs.keys():
#    m_V = Vs[key]
#    lon_mean = np.mean(m_V[:,LON])
#    lat_mean = np.mean(m_V[:,LAT])
#    if lon_mean == 1:
#        lon_mean = 0.99999
#    if lat_mean == 1:
#        lat_mean = 0.99999
#    Tiles[str(int(lat_mean*10))+str((int(lon_mean*10)))].append(key)
#
#v_density = np.empty((100,))
#for d_i in range(100):
#    key = "{0:02d}".format(d_i)
#    v_density[d_i] = len(Tiles[key])    
#plt.bar(range(100),v_density)
#plt.xlabel("Tile (lat+lon)")
#plt.ylabel("Density (unnormalised)")
#plt.title("Dataset2")
#
#d_density_max = 2500
#for d_i in range(100):
#    key_Tiles = "{0:02d}".format(d_i)
#    if len(Tiles[key_Tiles]) > d_density_max:
#        for key_Vs in Tiles[key_Tiles][d_density_max:]:
#            Vs.pop(key_Vs,None)
            

# Step 7bis:Train-test splitting
###############################################################################
print('Train-test splitting...')

Vs = Data
v_all_idx = np.random.permutation(len(Vs))
l_keys = Vs.keys()
Vs_train = dict()
Vs_valid = dict()
Vs_test = dict()
for d_i in v_all_idx[:int(len(Vs)*0.6)]:
    key = l_keys[d_i]
    Vs_train[key] = Vs[key]
for d_i in v_all_idx[int(len(Vs)*0.6):int(len(Vs)*0.9)]:
    key = l_keys[d_i]
    Vs_valid[key] = Vs[key]
for d_i in v_all_idx[int(len(Vs)*0.9):]:
    key = l_keys[d_i]
    Vs_test[key] = Vs[key]
    
with open("/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/dataset8/dataset8_train.pkl","wb") as f:
    pickle.dump(Vs_train,f)
with open("/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/dataset8/dataset8_valid.pkl","wb") as f:
    pickle.dump(Vs_valid,f)
with open("/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/dataset8/dataset8_test.pkl","wb") as f:
    pickle.dump(Vs_test,f)
