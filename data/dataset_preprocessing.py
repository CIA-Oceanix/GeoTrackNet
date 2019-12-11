# coding: utf-8

# MIT License
# 
# Copyright (c) 2018 Duong Nguyen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

"""
A script to prepare data for GeoTrackNet.
"""

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
from tqdm import tqdm

## PARAMS
#======================================

CARGO_TANKER_ONLY = True

## Gulf of Mexico
"""
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
"""

## Bretagne
LAT_MIN = 47.5
LAT_MAX = 49.5
LON_MIN = -7.0
LON_MAX = -4.0

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

## data path
dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/"
filename_list = ["ct_2017070809_10_20_valid_track.pkl"]
pkl_filepath = "./ct_2017070809_10_20/2017070809_10_20_valid.pkl"

## Time
t_min = time.mktime(time.strptime("01/07/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("30/09/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))


dict_list = []
for filename in filename_list:
    with open(os.path.join(dataset_path,filename),"rb") as f:
        temp = pickle.load(f)
        dict_list.append(temp)


## MarineC
"""
data_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/"
dict_list = [] # List of data dictionary
month = "01"
zone_list = ['14','15','16']
filename_list = []
for zone in zone_list:
    filename = data_path + month + "/Zone" + zone + "_2014_" + month + ".pkl"
    filename_list.append(filename)
    print("Loading ", filename, "...")
    with open(filename,"rb") as f:
        temp = pickle.load(f)
        dict_list.append(temp)
"""


## Uncomment if you want to create shapefile
#for Vi,zone in zip(dict_list, zone_list):
#    filename = data_path + month + "/Zone" + zone + "_2014_01.shp"
#    print("Creating " + filename + "...")
#    utils.createShapefile(filename,Vi)


## STEP1: FILTERING
#======================================
# Remove erroneous timestamps and erroneous speeds, then merge zones
print(" Remove erroneous timestamps and erroneous speeds...")
Vs = dict()
for Vi,filename in zip(dict_list, filename_list):
    print(filename)
    for mmsi in list(Vi.keys()):
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
        if mmsi not in list(Vs.keys()):
            Vs[mmsi] = Vi[mmsi]
            del Vi[mmsi]
        else:
            Vs[mmsi] = np.concatenate((Vs[mmsi],Vi[mmsi]),axis = 0)
            del Vi[mmsi]
del dict_list, Vi, abnormal_speed_idx, abnormal_timestamp_idx



## STEP 2: VOYAGES SPLITTING 
#======================================
# Cutting discontiguous voyages into contiguous ones
print("Cutting discontiguous voyages into contiguous ones...")
count = 0
voyages = dict()
INTERVAL_MAX = 4*3600 # 2h
for mmsi in list(Vs.keys()):
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



# STEP 3: REMOVING SHORT VOYAGES
#======================================
# Removing AIS track whose length is smaller than 20 or those last less than 4h
print("Removing AIS track whose length is smaller than 20 or those last less than 4h...")

for mmsi in list(voyages.keys()):
    duration = voyages[mmsi][-1,TIMESTAMP] - voyages[mmsi][0,TIMESTAMP]
    if (len(voyages[mmsi]) < 20) or (duration < 4*3600):
        voyages.pop(mmsi, None)


# STEP 4: REMOVING OUTLIERS
#======================================
print("Removing anomalous message...")
error_count = 0
tick = time.time()
for mmsi in  tqdm(list(voyages.keys())):
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
print("STEP 4: duration = ",(tok - tick)/60) # 139.685766101 mfrom tqdm import tqdmins



## STEP 5: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
#======================================
# Removing 'moored' or 'at anchor' voyages
print("Removing 'moored' or 'at anchor' voyages...")
for mmsi in  tqdm(list(voyages.keys())):
    d_L = float(len(voyages[mmsi]))

    if np.count_nonzero(voyages[mmsi][:,NAV_STT] == 1)/d_L > 0.7\
       or np.count_nonzero(voyages[mmsi][:,NAV_STT] == 5)/d_L > 0.7:
        voyages.pop(mmsi,None)
        continue
    sog_max = np.max(voyages[mmsi][:,SOG])
    if sog_max < 1.0:
        voyages.pop(mmsi,None)



## STEP 6: SAMPLING
#======================================
# Sampling, resolution = 5 min
print('Sampling...')
Vs = dict()
count = 0
for k in tqdm(list(voyages.keys())):
    v = voyages[k]
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

## STEP 7: REMOVING LOW SPEED TRACKS
#======================================
print("Removing 'low speed' tracks...")
for mmsi in tqdm(list(Vs.keys())):
    d_L = float(len(Vs[mmsi]))
    if np.count_nonzero(Vs[mmsi][:,SOG] < 2)/d_L > 0.8:
        Vs.pop(mmsi,None)

## STEP 8: RE-SPLITTING
#======================================
print('Re-Splitting...')
Data = dict()
count = 0
for mmsi in tqdm(list(Vs.keys())): 
    v = Vs[mmsi]
    # Split AIS track into small tracks whose duration <= 1 day
    idx = np.arange(0, len(v), 12*24)[1:]
    tmp = np.split(v,idx)
    for subtrack in tmp:
        # only use tracks whose duration >= 4 hours
        if len(subtrack) > 12*4:
            Data[count] = subtrack
            count += 1


## STEP 9: NORMALISATION
#======================================
print('Normalisation...')
for k in tqdm(list(Data.keys())):
    v = Data[k]
    v[:,LAT] = (v[:,LAT] - LAT_MIN)/(LAT_MAX-LAT_MIN)
    v[:,LON] = (v[:,LON] - LON_MIN)/(LON_MAX-LON_MIN)
    v[:,SOG][v[:,SOG] > SPEED_MAX] = SPEED_MAX
    v[:,SOG] = v[:,SOG]/SPEED_MAX
    v[:,COG] = v[:,COG]/360.0


## STEP 10: WRITING TO DISK
#======================================
with open(pkl_filepath,"wb") as f:
    pickle.dump(Data,f)


"""
with open("/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/ct_010203_24/ct_010203_24_train.pkl","rb") as f:
    Vs = pickle.load(f)

plt.figure()
for key in Vs.keys():
    v = Vs[key]
    plt.plot(v[:,LON],v[:,LAT])
"""
## In[]
### Step 11: Density normalisation
###############################################################################
#with open("/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/ct_010203_24/ct_010203_24_train.pkl","rb") as f:
#    Vs = pickle.load(f)
#
#Tiles = dict()
#for d_i in range(10):
#    for d_j in range(10):
#        Tiles[str(d_i)+str(d_j)] = []
#for key in list(Vs.keys()):
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
#plt.bar(list(range(100)),v_density)
#plt.xlabel("Tile (lat+lon)")
#plt.ylabel("Density (unnormalised)")
#plt.title("Dataset2")
#
#d_density_max = 1000
#for d_i in range(100):
#    key_Tiles = "{0:02d}".format(d_i)
#    if len(Tiles[key_Tiles]) > d_density_max:
#        for key_Vs in Tiles[key_Tiles][d_density_max:]:
#            Vs.pop(key_Vs,None)

## Step 11bis:Train-test splitting
################################################################################
#print('Train-test splitting...')
#
##Vs = Data
#v_all_idx = np.random.permutation(len(Vs))
#l_keys = list(Vs.keys())
#Vs_train = dict()
#Vs_valid = dict()
#Vs_test = dict()
#for d_i in v_all_idx[:int(len(Vs)*0.6)]:
#    key = l_keys[d_i]
#    Vs_train[key] = Vs[key]
#for d_i in v_all_idx[int(len(Vs)*0.6):int(len(Vs)*0.9)]:
#    key = l_keys[d_i]
#    Vs_valid[key] = Vs[key]
#for d_i in v_all_idx[int(len(Vs)*0.9):]:
#    key = l_keys[d_i]
#    Vs_test[key] = Vs[key]
#
#with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/MarineC_Jan2014_Norm/MarineC_Jan2014_Norm_train.pkl","wb") as f:
#    pickle.dump(Vs_train,f)
#with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/MarineC_Jan2014_Norm/MarineC_Jan2014_Norm_valid.pkl","wb") as f:
#    pickle.dump(Vs_valid,f)
#with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/MarineC_Jan2014_Norm/MarineC_Jan2014_Norm_test.pkl","wb") as f:
#    pickle.dump(Vs_test,f)
