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
A script to merge AIS messages into AIS tracks.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
import pickle
import matplotlib.pyplot as plt
import copy
import csv
from datetime import datetime
import time
from io import StringIO

## PARAMS
#======================================

## Gulf of Mexico
#LAT_MIN = 26.5
#LAT_MAX = 30.0
#LON_MIN = -97.5
#LON_MAX = -87

## Brittany
LAT_MIN = 47.5
LAT_MAX = 49.5
LON_MIN = -7.0
LON_MAX = -4.0

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # the SOG is truncated to 30.0 knots max.

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

## Brittany

# Path to csv files.
dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/"
l_csv_filename =["01_position.csv","02_position.csv","03_position.csv"]


# Pkl filenames.
pkl_filename = "010203_track.pkl"
pkl_filename_train = "010203_10_20_train_track.pkl"
pkl_filename_valid = "010203_10_20_valid_track.pkl"
pkl_filename_test  = "010203_10_20_test_track.pkl"

CARGO_TANKER_ONLY = True
if  CARGO_TANKER_ONLY:
    pkl_filename += "ct_"
    pkl_filename_train += "ct_"
    pkl_filename_valid += "ct_"
    pkl_filename_test  += "ct_"


# Training/validation/test/total period.
t_train_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_train_max = time.mktime(time.strptime("10/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_valid_min = time.mktime(time.strptime("11/03/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_valid_max = time.mktime(time.strptime("20/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_test_min  = time.mktime(time.strptime("21/03/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_test_max  = time.mktime(time.strptime("31/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))

# List of cargo or tanker vessels.
l_cargo_tanker = np.load("/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/010203_cargo_tanker.npy")


## LOADING CSV FILES
#======================================
l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)

for csv_filename in l_csv_filename:
    data_path = os.path.join(dataset_path,csv_filename)
    with open(data_path,"r") as f:
        print("Reading ", csv_filename, "...")
        csvReader = csv.reader(f)
        next(csvReader) # skip the legend row
        for row in csvReader:
            utc_time = datetime.strptime(row[7], "%Y/%m/%d %H:%M:%S")
            timestamp = (utc_time - EPOCH).total_seconds()
            l_l_msg.append([float(row[1]),float(row[0]),
                           float(row[3]),float(row[5]),
                           int(row[4]),0,
                           int(row[6]),int(timestamp),
                           int(row[2])])

"""
## MarineC
dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/"
for month in range(1,2):
    for zone in [14,15,16]:
        csv_filename = dataset_path + "{0:02d}/Zone{1:02d}_2014_{0:02d}.csv".format(month,zone)
        with open(csv_filename, 'r') as f:
            print("Reading ", csv_filename, "...")
            csvReader = csv.reader(f)
            next(csvReader) # skip the legend row
            for row in csvReader :
                lat = float(row[1])
                lon = float(row[0])
                if lat > LAT_MAX or lat < LAT_MIN or lon > LON_MAX or lon < LON_MIN:
                    continue
                utc_time = datetime.strptime(row[6], "%Y/%m/%d %H:%M:%S")
                timestamp = (utc_time - EPOCH).total_seconds()
                l_l_msg.append([lat, lon,
                             float(row[2]), float(row[3]),
                             float(row[4]), int(row[5]),
                             int(row[7]),
                             int(timestamp), int(row[9])])
"""

m_msg = np.array(l_l_msg)
del l_l_msg
print("Total number of AIS messages: ",m_msg.shape[0])

## FILTERING 
#======================================
# Selecting AIS messages in the ROI and in the period of interest.

## LAT LON
m_msg = m_msg[m_msg[:,LAT]>=LAT_MIN]
m_msg = m_msg[m_msg[:,LAT]<=LAT_MAX]
m_msg = m_msg[m_msg[:,LON]>=LON_MIN]
m_msg = m_msg[m_msg[:,LON]<=LON_MAX]
# SOG
m_msg = m_msg[m_msg[:,SOG]>=0]
m_msg = m_msg[m_msg[:,SOG]<=SOG_MAX]
# COG
m_msg = m_msg[m_msg[:,SOG]>=0]
m_msg = m_msg[m_msg[:,COG]<=360]
# TIME
m_msg = m_msg[m_msg[:,TIMESTAMP]>=0]

m_msg = m_msg[m_msg[:,TIMESTAMP]>=t_min]
m_msg = m_msg[m_msg[:,TIMESTAMP]<=t_max]
m_msg_train = m_msg[m_msg[:,TIMESTAMP]>=t_train_min]
m_msg_train = m_msg_train[m_msg_train[:,TIMESTAMP]<=t_train_max]
m_msg_valid = m_msg[m_msg[:,TIMESTAMP]>=t_valid_min]
m_msg_valid = m_msg_valid[m_msg_valid[:,TIMESTAMP]<=t_valid_max]
m_msg_test  = m_msg[m_msg[:,TIMESTAMP]>=t_test_min]
m_msg_test  = m_msg_test[m_msg_test[:,TIMESTAMP]<=t_test_max]



## MERGING INTO DICT
#======================================
# Creating AIS tracks from the list of AIS messages.
# Each AIS track is formatted by a dictionary.
print("Convert to dicts of vessel's tracks...")

## All AIS messages
#Vs = dict()
#for v_msg in m_msg:
#    mmsi = int(v_msg[MMSI])
#    if not (mmsi in list(Vs.keys())):
#        Vs[mmsi] = np.empty((0,9))
#    Vs[mmsi] = np.concatenate((Vs[mmsi], np.expand_dims(v_msg,0)), axis = 0)
#for key in list(Vs.keys()):
#    Vs[key] = np.array(sorted(Vs[key], key=lambda m_entry: m_entry[TIMESTAMP]))

#with open(os.path.join(dataset_path,pkl_filename),"wb") as f:
#    pickle.dump(Vs,f)

# Training set
Vs_train = dict()
for v_msg in m_msg_train:
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_train.keys())):
        Vs_train[mmsi] = np.empty((0,9))
    Vs_train[mmsi] = np.concatenate((Vs_train[mmsi], np.expand_dims(v_msg,0)), axis = 0)
for key in list(Vs_train.keys()):
    Vs_train[key] = np.array(sorted(Vs_train[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Validation set
Vs_valid = dict()
for v_msg in m_msg_valid:
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_valid.keys())):
        Vs_valid[mmsi] = np.empty((0,9))
    Vs_valid[mmsi] = np.concatenate((Vs_valid[mmsi], np.expand_dims(v_msg,0)), axis = 0)
for key in list(Vs_valid.keys()):
    Vs_valid[key] = np.array(sorted(Vs_valid[key], key=lambda m_entry: m_entry[TIMESTAMP]))

# Test set
Vs_test = dict()
for v_msg in m_msg_test:
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs_test.keys())):
        Vs_test[mmsi] = np.empty((0,9))
    Vs_test[mmsi] = np.concatenate((Vs_test[mmsi], np.expand_dims(v_msg,0)), axis = 0)
for key in list(Vs_test.keys()):
    Vs_test[key] = np.array(sorted(Vs_test[key], key=lambda m_entry: m_entry[TIMESTAMP]))


## PICKLING
#======================================
for filename, filedict in zip([pkl_filename_train,pkl_filename_valid,pkl_filename_test],
                              [Vs_train,Vs_valid,Vs_test]
                             ):
    print("Writing to ", os.path.join(dataset_path,filename),"...")
    print("Total number of tracks: ", len(filedict))
    
    with open(os.path.join(dataset_path,filename),"wb") as f:
        pickle.dump(filedict,f)
        

