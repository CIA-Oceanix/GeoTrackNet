#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:13:07 2018

@author: vnguye04

Dataset Visualisation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import sys
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)
# AMERICA
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots



## TRAJECTORIES
###############################################################################

dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/dataset3/dataset3_test.pkl"
dataset_name = os.path.basename(dataset_path).split('.')[0]
with open(dataset_path,"rb") as f:
    Vs = pickle.load(f)

l_keys = Vs.keys()
num_samples = len(Vs)
v_idx = np.random.permutation(num_samples)
count = 0
for d_i in v_idx[0:int(num_samples)]:
    count += 1
    print(count)
    m_V = Vs[l_keys[d_i]]
    v_lon = m_V[:,LON]*LON_RANGE + LON_MIN
    v_lat = m_V[:,LAT]*LAT_RANGE + LAT_MIN
    plt.plot(v_lon,v_lat)

plt.title("Visualisation of {0} AIS tracks in the test set ({1})".format(num_samples,dataset_name))
plt.xlabel("Longitude (normalized)")
plt.ylabel("Latitude (normalized)")
plt.xlim([LON_MIN,LON_MAX])
plt.ylim([LAT_MIN,LAT_MAX])
plt.show()


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



## FISHING VESSELS
###############################################################################

dataset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/Data_Zone141516_normalized_5_train.pkl"
#dataset_path = dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_2_train.pkl"
with open(dataset_path,"rb") as f:
    Vs = pickle.load(f)

v_keys = Vs.keys()
#num_samples = len(Vs)
num_samples = len(Vs)
v_idx = np.random.permutation(num_samples)
count = 0
for d_i in v_idx:
    count += 1
    print(count)
    m_V = Vs[v_keys[d_i]]
    mmsi = m_V[0,MMSI]
    try:
        if m_vessel_type[m_vessel_type[:,0]==mmsi][0,1] == 30:
            plt.plot(m_V[:,LON],m_V[:,LAT],'r')
        else:
            plt.plot(m_V[:,LON],m_V[:,LAT],'b')
    except:
        continue

#plt.title("Visualisation of {} AIS tracks in the training set".format(num_samples))
plt.ylim([0,1])
plt.title("Fishing vessel")
plt.xlabel("Longitude (normalized)")
plt.ylabel("Latitude (normalized)")
plt.show()


### TRAIN TEST SPLITTING
###############################################################################
"""
dataset_path = dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_3.pkl"

with open(dataset_path,"rb") as f:
    Vs = pickle.load(f)

num_samples = len(Vs)
v_idx_all = np.random.permutation(num_samples)
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_4_idx_all.pkl","wb") as f:
    pickle.dump(v_idx_all,f)
num_train = int(num_samples*0.8)
v_keys = np.array(Vs.keys())
Vs_train = dict()
d_count = 0
for d_i in v_idx_all[0:num_train]:
    d_count += 1
    print(d_count)
    key = v_keys[d_i]
    Vs_train[key] = Vs[key]
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_4_train.pkl","wb") as f:
    pickle.dump(Vs_train,f)
    
Vs_test = dict()
d_count = 0
for d_i in v_idx_all[num_train:]:
    d_count += 1
    print(d_count)
    key = v_keys[d_i]
    Vs_test[key] = Vs[key]
with open("/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_4_test.pkl","wb") as f:
    pickle.dump(Vs_test,f)
"""


## NAVIGATION STATUS
###############################################################################

"""
dataset_path = dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_3_train.pkl"
#dataset_path = dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_2_train.pkl"
with open(dataset_path,"rb") as f:
    Vs = pickle.load(f)
v_keys = Vs.keys()

m_nav_stt = []
v_nav_stt = np.empty((0,))
for k in v_keys:
    m_nav_stt.append(Vs[k][:,NAV_STT])
    v_nav_stt = np.concatenate((v_nav_stt,Vs[k][:,NAV_STT]))
stt_bins = plt.hist(v_nav_stt, bins = (np.arange(0,17) - 0.5))[0]
plt.xlabel("Navigation status")
plt.title("US dataset 2")
"""

## FISHING
###############################################################################
"""
dataset_path = dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_2_train.pkl"
with open(dataset_path,"rb") as f:
    Vs = pickle.load(f)
v_keys = Vs.keys()
Fishings = dict()
for k in v_keys:
    if (Vs[k][:,NAV_STT] == 7).any():
        Fishings[k] = np.copy(Vs[k])
    
for k in Vs.keys():
    tmp = Vs[k]
    if (Vs[k][:,NAV_STT] == 7).any():
        plt.plot(tmp[:,1],tmp[:,0],'r')
    else:
        plt.plot(tmp[:,1],tmp[:,0],'b')
plt.xlabel("Latitude (normalized)")
plt.ylabel("Longitude (normalized)")
plt.title("Fishing vessel (red) (training set 2)")
"""

## ZERO SPEED
###############################################################################
"""
dataset_path = dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Data_Zone141516_normalized_2_train.pkl"
with open(dataset_path,"rb") as f:
    Vs = pickle.load(f)
v_keys = Vs.keys()
m_sogs = []
v_sogs = np.empty((0,))
count = 0
for k in v_keys:
    count += 1
    print(count)
    m_sogs.append(Vs[k][:,SOG])
    v_sogs = np.concatenate((v_sogs,Vs[k][:,SOG]))
    
for d_k in Vs.keys():
    v_sogs = Vs[d_k][:,SOG]
    if float(np.count_nonzero(v_sogs < 2/12.))/len(v_sogs) > 0.8:
        plt.plot(Vs[d_k][:,LON],Vs[d_k][:,LAT])

plt.xlabel("Latitude (normalized)")
plt.ylabel("Longitude (normalized)")
plt.title("Low speed vessels")     
"""
    
## FISHING VESSEL 
###############################################################################
import csv
import numpy as np
import matplotlib.pyplot as plt
l_zone = [14,15,16]
dataset_vessel_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Zone{}_2014_01_Vessel.csv"
VesselTypes = dict()
v_vessel_types = []
for zone in l_zone:
    with open(dataset_vessel_path.format(zone), "rb") as f:
        reader = csv.reader(f)
        v_labels = reader.next()
        for row in reader:
            try:
                mmsi_ = int(row[0])
                type_ = int(row[4])
                v_vessel_types.append([mmsi_, type_])
                if  mmsi_ not in VesselTypes.keys():
                    VesselTypes[mmsi_] = [type_]
                elif type_ not in VesselTypes[mmsi_]:
                    VesselTypes[mmsi_].append(type_)
            except:
                continue
v_vessel_types = np.array(v_vessel_types).astype(np.int)
for mmsi_ in VesselTypes.keys():
    VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])

np.count_nonzero(np.logical_and(v_vessel_types[:,1] >= 80, v_vessel_types[:,1] <= 89))
np.count_nonzero(v_vessel_types[:,1] == 31)

def sublist(lst1, lst2):
   ls1 = [element for element in lst1 if element in lst2]
   ls2 = [element for element in lst2 if element in lst1]
   return (len(ls1) != 0) and (ls1 == ls2)
# Fishing: 91
# Tug 31: 2053
# Tug 32: 268
# Tug 52: 851
# Passenger 6x: 298
# Cargo 7x: 2006
# Tanker 8x: 1244


with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset0/dataset0_train.pkl","rb") as f:
    Vs_train = pickle.load(f)
    
Vs = Vs_train
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)

FIG_DPI = 150
plt.figure(figsize=(1920/FIG_DPI, 1080/FIG_DPI), dpi=FIG_DPI)
count = 0
for key_ in Vs.keys():
    count += 1
    print(count)
    tmp = Vs[key_]
    mmsi_ = int(tmp[0,MMSI])
    try:
        if sublist(VesselTypes[mmsi_], [30]): # fishing
            plt.plot(tmp[:,1],tmp[:,0],'r')
            pass
#        elif sublist(VesselTypes[mmsi_], [31,32,52]): # tug
#            plt.plot(tmp[:,1],tmp[:,0],'g')
#        elif sublist(VesselTypes[mmsi_], range(60,70)): # passenger
#            plt.plot(tmp[:,1],tmp[:,0],'b')
#        elif sublist(VesselTypes[mmsi_], range(70,80)): # cargo
#            plt.plot(tmp[:,1],tmp[:,0],'k')
        elif sublist(VesselTypes[mmsi_], range(80,90)): # tanker
            plt.plot(tmp[:,1],tmp[:,0],'m')
    except:
        continue
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Tankers")
plt.ylim([0,1])
plt.xlim([0,1])
plt.savefig("data/dataset0/Tankers.png",dpi = FIG_DPI)





###############################################################################
import numpy as np
import sys
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)
# AMERICA
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset3/dataset3_test.pkl","rb") as f:
    Vs_test = pickle.load(f)
    
Vs = Vs_test
count = 0
l_durations = []
for key in Vs.keys():
    print(count)
    count += 1
    tmp = Vs[key]
    l_durations.append(len(tmp)/12)
plt.hist(l_durations)
plt.title("Test set")
plt.xlabel("Tracks\' duration (hour)")
plt.show()











###############################################################################
import numpy as np
import sys
import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt


LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)
# AMERICA
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset4/dataset4_test.pkl","rb") as f:
    Vs_test = pickle.load(f)
    
Vs = Vs_test
count = 0
for key in Vs.keys():
    print(count)
    count += 1
    tmp = Vs[key]
    mmsi = int(tmp[0,MMSI])
    if mmsi == 366190838:
        plt.plot(tmp[:,1],tmp[:,0])
plt.ylim([0,1])
plt.xlim([0,1])
plt.show()
