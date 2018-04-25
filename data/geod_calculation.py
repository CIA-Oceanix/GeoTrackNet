#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:09:27 2018

@author: vnguye04
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import radians, cos, sin, asin, sqrt
import sys
import pickle
import shapefile
import time
from pyproj import Geod
geod = Geod(ellps='WGS84')
#import dataset

AVG_EARTH_RADIUS = 6378.137  # in km
SPEED_MAX = 30 # knot


z1, e1, n1 = 17, 630084.0, 4833438.0
z2, e2, n2 = 17, 660084.0, 4833438.0
lat1, lon1 = 43.642561, -79.387143
lat2, lon2 = 43.63671, -79.015385
eps = 1e-10
az_fw, az_bw, dist = geod.inv(lon1, lat1, lon2, lat2)

print("dist: ", dist, "az_fw: ", az_fw, "az_bw", az_bw)

dist_utm = np.sqrt((e1-e2)**2 + (n1-n2)**2)
az_utm = np.arctan((e2-e1)/(n2-n1 + eps))/np.pi*180
print("dist_utm: ", dist_utm, "az_utm: ", az_utm)