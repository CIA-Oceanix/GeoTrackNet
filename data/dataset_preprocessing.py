#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:13:06 2018

@author: vnguye04
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
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

# AMERICA #lon : [-98, -86.8], lat : [26.16, 30]
## dataset0 and dataset 1
#LAT_MIN = 25.0
#LAT_MAX = 30.0
#LON_MIN = -98.0
#LON_MAX = -84.0

## dataset2
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87.0

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)

# DATA PATH

data_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/"
dict_list = [] # List of data dictionary
month_list = ["01","02"]
zone_list = ['14','15','16']
filename_list = []
for month in month_list:
    for zone in zone_list:
        filename = data_path + month + "/Zone" + zone + "_2014_"+month+".pkl"
        filename_list.append(filename)
        print("Loading ", filename, "...")
        with open(filename,"rb") as f:
            temp = pickle.load(f)
            dict_list.append(temp)

## Uncomment if you want to create shapefile
#for Vi,zone in zip(dict_list, zone_list):
#    filename = data_path + month + "/Zone" + zone + "_2014_01.shp"
#    print("Creating " + filename + "...")
#    utils.createShapefile(filename,Vi)
    
# REMOVING ABNORMAL TIMESTAMPS AND ABNORMAL SPEEDS AND MERGING ZONES
print("REMOVING ABNORMAL TIMESTAMPS AND ABNORMAL SPEEDS AND MERGING ZONES...")
print("CHANGING BOUNDARY (LAT, LON)...")
t_min = time.mktime(time.strptime("01/01/2014 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("28/02/2014 23:59:59", "%d/%m/%Y %H:%M:%S"))
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
              

#v_speed_max = [] 
#
#for mmsi in Vs.keys():
#    speed_max_tmp = np.max(Vs[mmsi][:,LON])
#    v_speed_max.append(speed_max_tmp)
#plt.hist(v_speed_max)

# STEP 1: Cutting discontinuous voyages into smaller voyages (5176 -> 48817)
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
            

# STEP 2: Removing AIS track whose length is smaller than 20 or who lasts less than 4h (48817 -> 26715)
###############################################################################
print("Removing AIS track whose length is smaller than 20 or who lasts less than 4h")
for mmsi in voyages.keys():
    duration = voyages[mmsi][-1,TIMESTAMP] - voyages[mmsi][0,TIMESTAMP]
    if (len(voyages[mmsi]) < 20) or (duration < 4*3600):
        voyages.pop(mmsi, None)
        

# STEP 3: Removing anomalous message (26715 -> 31274)
###############################################################################
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
        count += 1
tok = time.time()
print("STEP 3: duration = ",(tok - tick)/60) # 125.02797935 mins


with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/voyages0102_step3.pkl","wb") as f:
    pickle.dump(voyages,f) # 49676
#with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/voyages0102_step3.pkl","rb") as f:
#    voyages = pickle.load(f)



## STEP 4: Removing 'moored' or 'ar anchor' tracks (xxxx -> 39038) # 45796975 AIS messages in total
###############################################################################
print("Removing 'moored' or 'ar anchor' tracks")
for mmsi in voyages.keys():
    d_L = float(len(voyages[mmsi]))
    
    if np.count_nonzero(voyages[mmsi][:,NAV_STT] == 1)/d_L > 0.7\
       or np.count_nonzero(voyages[mmsi][:,NAV_STT] == 5)/d_L > 0.7:
        voyages.pop(mmsi,None)
        continue
    sog_max = np.max(voyages[mmsi][:,SOG])
    if sog_max < 1.0:
        voyages.pop(mmsi,None)


## Visualisation
#######################################
#l_keys = voyages.keys()
#num_samples = len(voyages)
#v_idx = np.random.permutation(num_samples)
#count = 0
#FIG_DPI = 150
#plt.figure(figsize=(1920/FIG_DPI, 686/FIG_DPI), dpi=FIG_DPI)
#for d_i in v_idx[0:int(num_samples/2)]:
#    count += 1
#    print(count)
#    m_V = voyages[l_keys[d_i]]
#    v_lon = m_V[:,LON]
#    v_lat = m_V[:,LAT]
#    plt.plot(v_lon,v_lat)
#
#dataset_name = "dataset1"
#plt.title("Visualisation of {0} AIS tracks in the dataset ({1})".format(int(num_samples/2),dataset_name))
#plt.xlabel("Longitude (normalized)")
#plt.ylabel("Latitude (normalized)")
#plt.xlim([LON_MIN,LON_MAX])
#plt.ylim([LAT_MIN,LAT_MAX])
#plt.xlim([-97.5,-87])
#plt.ylim([26.5,30])
#plt.show()



#no_msg = 0
#for mmsi in voyages.keys():
#    no_msg += len(voyages[mmsi])
#print("Total number of AIS messages:  ", no_msg)

# Step 5: Sampling (24285 -> 15681), resolution = 5 min
###############################################################################
tick = time.time()
print('STEP 5: Sampling...')
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
tok = time.time() #51.23


with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/voyages0102_step5.pkl","wb") as f:
    pickle.dump(Vs,f) #39036
    
#with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/voyages0102_step5.pkl","rb") as f:
#    voyages = pickle.load(f)

## (24285 -> 20566)

print("Removing 'low speed' tracks") #39036->25219
for mmsi in Vs.keys():
    d_L = float(len(Vs[mmsi]))    
    if np.count_nonzero(Vs[mmsi][:,SOG] < 2)/d_L > 0.8:
        Vs.pop(mmsi,None)
# (20566 -> 15681)

# STEP 6: Re-Splitting (25219-> 51796)
###############################################################################
print('STEP 6: Re-Splitting...')
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
## STATISTICS
#num_msgs = 0
#nav_stt = np.empty((0,))
#for mmsi in Data.keys():
#    num_msgs += len(Data[mmsi])
#    nav_stt = np.hstack((nav_stt,Data[mmsi][:,NAV_STT]))


# Step 7: Normalisation (51796)
###############################################################################
print('STEP 7: Normalisation...')
for k in Data.keys():
    v = Data[k]
    v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
    v[:,SOG] = v[:,SOG]/SPEED_MAX
    v[:,COG] = v[:,COG]/360.0
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/dataset2.pkl","wb") as f:
    pickle.dump(Data,f)
    
## Train-test splitting
v_all_idx = np.random.permutation(len(Data))
l_keys = Data.keys()
Vs_train = dict()
Vs_test = dict()
for d_i in v_all_idx[:int(len(Data)*0.8)]:
    key = l_keys[d_i]
    Vs_train[key] = Data[key]
for d_i in v_all_idx[int(len(Data)*0.8):]:
    key = l_keys[d_i]
    Vs_test[key] = Data[key]
    
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/dataset2_train.pkl","wb") as f:
    pickle.dump(Vs_train,f)
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/dataset2_test.pkl","wb") as f:
    pickle.dump(Vs_test,f)


# Step 8: Density normalisation
###############################################################################
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset2/dataset2.pkl","rb") as f:
    Vs = pickle.load(f)
    
Tiles = dict()
for d_i in range(10):
    for d_j in range(10):
        Tiles[str(d_i)+str(d_j)] = []   
for key in Vs.keys():
    m_V = Vs[key]
    lon_mean = np.mean(m_V[:,LON])
    lat_mean = np.mean(m_V[:,LAT])
    if lon_mean == 1:
        lon_mean = 0.99999
    if lat_mean == 1:
        lat_mean = 0.99999
    Tiles[str(int(lat_mean*10))+str((int(lon_mean*10)))].append(key)

v_density = np.empty((100,))
for d_i in range(100):
    key = "{0:02d}".format(d_i)
    v_density[d_i] = len(Tiles[key])    
plt.bar(range(100),v_density)
plt.xlabel("Tile (lat+lon)")
plt.ylabel("Density (unnormalised)")
plt.title("Dataset2")

for d_i in range(100):
    key_Tiles = "{0:02d}".format(d_i)
    if len(Tiles[key_Tiles]) > 1500:
        for key_Vs in Tiles[key_Tiles][1500:]:
            Vs.pop(key_Vs,None)
            
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3.pkl","wb") as f:
    pickle.dump(Vs,f)
    
## Train-test splitting
v_all_idx = np.random.permutation(len(Vs))
l_keys = Vs.keys()
Vs_train = dict()
Vs_test = dict()
for d_i in v_all_idx[:int(len(Vs)*0.8)]:
    key = l_keys[d_i]
    Vs_train[key] = Vs[key]
for d_i in v_all_idx[int(len(Vs)*0.8):]:
    key = l_keys[d_i]
    Vs_test[key] = Vs[key]
    
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_train.pkl","wb") as f:
    pickle.dump(Vs_train,f)
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_test.pkl","wb") as f:
    pickle.dump(Vs_test,f)
    

# Step 9: 1/2-day dataset 
###############################################################################
#os.makedirs("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/datasets4")
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_train.pkl","rb") as f:
    Vs_train = pickle.load(f)
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_test.pkl","rb") as f:
    Vs_test = pickle.load(f)

Vs = Vs_train
Vs = Vs_test
for key in Vs.keys():
    if len(Vs[key]) < 144:
        Vs.pop(key,None)
        
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset4/dataset4_train.pkl","wb") as f:
    pickle.dump(Vs,f)
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset4/dataset4_test.pkl","wb") as f:
    pickle.dump(Vs,f)


# Step 9: 8-hour dataset 
###############################################################################
#os.makedirs("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset5")
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_train.pkl","rb") as f:
    Vs_train = pickle.load(f)
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_test.pkl","rb") as f:
    Vs_test = pickle.load(f)

Vs = Vs_train
Vs = Vs_test
for key in Vs.keys():
    if len(Vs[key]) < 8*12:
        Vs.pop(key,None)
        
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset5/dataset5_train.pkl","wb") as f:
    pickle.dump(Vs,f)
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset5/dataset5_test.pkl","wb") as f:
    pickle.dump(Vs,f)
    
