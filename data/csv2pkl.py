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
from math import radians, cos, sin, asin, sqrt
import sys
import utils
import pickle
import matplotlib.pyplot as plt
import copy
import csv
from datetime import datetime
import time
from io import StringIO
#import utm


# AMERICA
LAT_MIN = 18.0
LAT_MAX = 30.0
LON_MIN = -98
LON_MAX = -84
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = range(9)

# DATA PATH
data_path = "/users/local/dnguyen/AIS_dataset/MarineC/2014/"

for month in range(1,5):
    for zone in [14,15,16]:
        csv_filename = data_path + "{0:02d}/Zone{1:02d}_2014_{0:02d}.csv".format(month,zone) # 61722685 msgs
        # data structure: 
        # [lon (0), lat, SOG (2), COG, Heading (4), ROT, DATETIME (6), status, VoyageID, MMSI (9), ....]
        data = []
        with open(csv_filename, 'r') as f:
            print("Reading ", csv_filename, "...")
            reader = csv.reader(f)
            reader.next() # skip the label row
            for row in reader:
                lat = float(row[1])
                lon = float(row[0])
                if lat > LAT_MAX or lat < LAT_MIN or lon > LON_MAX or lon < LON_MIN:
                    continue
                utc_time = datetime.strptime(row[6], "%Y/%m/%d %H:%M:%S")
                timestamp = (utc_time - EPOCH).total_seconds()
                data.append([lat, lon, 
                             float(row[2]), float(row[3]),
                             float(row[4]), int(row[5]), 
                             int(row[7]), 
                             int(timestamp), int(row[9])])
        print("Total number of AIS messages in the ROI: ", len(data))
        # Convert to dict of vessel's tracks
        print("Convert to dict of vessel's tracks...")
        Vs = dict()
        for msg in data:
            mmsi = msg[MMSI]
            if not (mmsi in Vs.keys()):
                Vs[mmsi] = np.empty((0,9))
            Vs[mmsi] = np.concatenate((Vs[mmsi], np.expand_dims(msg,0)), axis = 0)
        print("Pickling...")        
        with open(csv_filename.replace(".csv","2.pkl"), "wb") as f:
            pickle.dump(Vs,f)
        del Vs
    

#with open(csv_filename.replace("csv","pkl"), "wb") as f:
#    pickle.dump(data,f)

#speeds = np.array([x[SOG] for x in data])
#plt.hist(speeds[speeds > 0.01], bins = 100)


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