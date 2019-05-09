#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:50:45 2018

@author: vnguye04
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("..")
import utils
import pickle
import matplotlib.pyplot as plt
import copy
import csv
from datetime import datetime
import time
from io import StringIO
#import utm

## Gulf of Mexico
#LAT_MIN = 26.5
#LAT_MAX = 30.0
#LON_MIN = -97.5
#LON_MAX = -87

### Bretagne
LAT_MIN = 47.0
LAT_MAX = 50.0
LON_MIN = -7.0
LON_MAX = -4.0

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SOG_MAX = 30.0  # knots

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

# DATA PATH 
l_l_msg = [] # list of AIS messages, each row is a message (list of AIS attributes)

## Bretagne
dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/mt314/aivdm/2017/"
csv_filename = os.path.join(dataset_path,"010203_position.csv")

with open(csv_filename,"rb") as f:
    print("Reading ", csv_filename, "...")
    csvReader = csv.reader(f)
    csvReader.next() # skip the legend row
    for row in csvReader:
        utc_time = datetime.strptime(row[7], "%Y/%m/%d %H:%M:%S")
        timestamp = (utc_time - EPOCH).total_seconds()
        l_l_msg.append([float(row[1]),float(row[0]),
                       float(row[3]),float(row[5]),
                       int(row[4]),0,
                       int(row[6]),int(timestamp),
                       int(row[2])])


n, bins, patches = plt.hist(m_msg[:,LAT],bins=44+np.arange(13)/2,cumulative=True)
np.count_nonzero(m_msg[:,LAT]>47)/float(len(m_msg))
np.count_nonzero(m_msg[:,LAT]<50)/float(len(m_msg))

n, bins, patches = plt.hist(m_msg[:,LON],bins=-7+np.arange(13)/5,cumulative=True)
np.count_nonzero(m_msg[:,LON]>-7)/float(len(m_msg))
np.count_nonzero(m_msg[:,LON]<-4)/float(len(m_msg))
np.count_nonzero(m_msg[:,SOG]<0)/float(len(m_msg))

np.count_nonzero(m_msg[:,SOG]>30)/float(len(m_msg))
    
## MarineC
"""
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
m_msg = np.array(l_l_msg)        



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
timestamp_max = (datetime(2017, 0o1, 31, 23, 59, 59) - EPOCH).total_seconds()
timestamp_max = (datetime(2017, 0o3, 31, 23, 59, 59) - EPOCH).total_seconds()
m_msg = m_msg[m_msg[:,TIMESTAMP]<=timestamp_max]

print("Convert to dict of vessel's tracks...")
Vs = dict()
for v_msg in m_msg:
    mmsi = int(v_msg[MMSI])
    if not (mmsi in list(Vs.keys()):
        Vs[mmsi] = np.empty((0,9))
    Vs[mmsi] = np.concatenate((Vs[mmsi], np.expand_dims(v_msg,0)), axis = 0)
for key in list(Vs.keys()):
    Vs[key] = np.array(sorted(Vs[key], key=lambda m_entry: m_entry[TIMESTAMP]))
    
#for key in Vs.keys():
#    tmp = Vs[key]
#    plt.plot(tmp[:,LON],tmp[:,LAT])
"""


print("Pickling...") 
## Bretagne       
with open(os.path.join(dataset_path,"010203_position.pkl"),"wb") as f:
    pickle.dump(Vs,f)

### MarineC
"""
with open(os.path.join(dataset_path,"01_position.pkl"),"wb") as f:
    pickle.dump(Vs,f)
"""
            
#cmap = plt.cm.get_cmap('Blues')



"""
VISUALISATION
"""
#csv_filename = data_path + "01/Zone15_2014_01.pkl"
#with open(csv_filename,"rb") as f:
#    Vs = pickle.load(f)
#
## REMOVING ABNORMAL TIMESTAMPS AND ABNORMAL SPEEDS
#t_min = time.mktime(time.strptime("01/01/2014 00:00:00", "%d/%m/%Y %H:%M:%S"))
#t_max = time.mktime(time.strptime("31/01/2014 23:59:59", "%d/%m/%Y %H:%M:%S"))
#for mmsi in Vs.keys():
#    # Abnormal timestamps
#    abnormal_timestamp_idx = np.logical_or((Vs[mmsi][:,TIMESTAMP] > t_max),
#                                           (Vs[mmsi][:,TIMESTAMP] < t_min))
#    Vs[mmsi] = Vs[mmsi][np.logical_not(abnormal_timestamp_idx)]
#    # Abnormal speeds
#    abnormal_speed_idx = Vs[mmsi][:,SOG] > SPEED_MAX
#    Vs[mmsi] = Vs[mmsi][np.logical_not(abnormal_speed_idx)]
#    # Deleting empty keys
#    if len(Vs[mmsi]) == 0:
#        del Vs[mmsi]
#        
#
## INTERVALS
#intervals = np.empty((0,))
#durations = []
#num_msgs = []
#for mmsi in Vs.keys():
#    # Number of AIS messages
#    num_msgs.append([mmsi, len(Vs[mmsi])])
#    # Duration of each vessel's track
#    durations.append([mmsi, Vs[mmsi][-1,TIMESTAMP] - Vs[mmsi][0,TIMESTAMP]])
#    # Intervals
#    intervals = np.concatenate((intervals,Vs[mmsi][1:,TIMESTAMP] - Vs[mmsi][:-1,TIMESTAMP]),axis = 0)
#
#num_msgs = np.array(num_msgs)    
#plt.hist(num_msgs[:,1], bins = 100)
#plt.title("Zone15_2014_01")
#plt.xlabel("Number of AIS messages sent by each vessel")    
#
#durations = np.array(durations)
#durations[:,1] = durations[:,1]/3600
#plt.hist(durations[:,1], bins = 100)
#plt.title("Zone15_2014_01")
#plt.xlabel("Duration of each vessel's track (hour)")  
#
#plt.hist(intervals, range = (0,2000), bins = 100, log = False)
#plt.title("Zone15_2014_01")
#plt.xlabel("Interval between two consecutive AIS messages (second)")  
