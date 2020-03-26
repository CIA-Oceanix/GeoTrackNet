
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
import sys
import os
from tqdm import tqdm_notebook as tqdm
try:
    sys.path.remove('/sanssauvegarde/homes/vnguye04/Codes/DAPPER')
except:
    pass
sys.path.append("..")
import utils
import pickle
import matplotlib.pyplot as plt
import copy
from datetime import datetime
import time
from io import StringIO

from tqdm import tqdm
import argparse

# In[2]:
def getConfig(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")
    # ROI
    parser.add_argument("--lat_min", type=float, default=9.0,
                        help="Lat min.")
    parser.add_argument("--lat_max", type=float, default=14.0,
                        help="Lat max.")
    parser.add_argument("--lon_min", type=float, default=-71.0,
                        help="Lon min.")
    parser.add_argument("--lon_max", type=float, default=-66.0,
                        help="Lon max.")
     
    # File paths
    parser.add_argument("--dataset_dir", type=str, 
                        default="./",
                        help="Dir to dataset.")    
    parser.add_argument("--l_input_filepath", type=str, nargs='+',
                        default=["ct_aruba_20172020_summer_valid_track.pkl"],
                        help="List of path to input files.")
    parser.add_argument("--output_filepath", type=str,
                        default="./ct_aruba_20172020_summer/ct_aruba_20172020_summer_valid.pkl",
                        help="Path to output file.")
    
    parser.add_argument("-v", "--verbose",dest='verbose',action='store_true', help="Verbose mode.")
    config = parser.parse_args(args)
    return config

config = getConfig(sys.argv[1:])

## Bretagne dataset
# LAT_MIN = 46.5
# LAT_MAX = 50.5
# LON_MIN = -8.0
# LON_MAX = -3.0
"""
LAT_MIN = 45.5
LAT_MAX = 50.5
LON_MIN = -10.0
LON_MAX = -1.0

FIG_W = 960
FIG_H = 533 #768

config.l_input_filepath = ["ct_bretagneB_20170103_10_20_valid_track.pkl"]
config.output_filepath = "./ct_bretagneB_20170103_10_20/ct_bretagneB_20170103_10_20_valid.pkl"

t_min = time.mktime(time.strptime("01/01/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("31/03/2017 23:59:59", "%d/%m/%Y %H:%M:%S"))
"""
# ## Aruba
"""
LAT_MIN = 9.0
LAT_MAX = 14.0
LON_MIN = -71.0
LON_MAX = -66.0

FIG_W = 960
FIG_H = int(960*5/5) #533 #768

config.l_input_filepath = ["ct_aruba_20172020_summer_valid_track.pkl"]
config.output_filepath = "./ct_aruba_20172020_summer/ct_aruba_20172020_summer_valid.pkl"

# t_min = time.mktime(time.strptime("01/11/2017 00:00:00", "%d/%m/%Y %H:%M:%S"))
# t_max = time.mktime(time.strptime("31/01/2020 23:59:59", "%d/%m/%Y %H:%M:%S"))
"""

"""
LAT_MIN = 9.0
LAT_MAX = 14.0
LON_MIN = -71.0
LON_MAX = -66.0



config.l_input_filepath = ["ct_aruba_2019_summer_valid_track.pkl"]
config.output_filepath = "./ct_aruba_2019_summer/ct_aruba_2019_summer_valid.pkl"

t_min = time.mktime(time.strptime("01/05/2019 00:00:00", "%d/%m/%Y %H:%M:%S"))
t_max = time.mktime(time.strptime("30/06/2019 23:59:59", "%d/%m/%Y %H:%M:%S"))
"""
#=====================================================================
LAT_MIN,LAT_MAX,LON_MIN,LON_MAX = config.lat_min,config.lat_max,config.lon_min,config.lon_max

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
DURATION_MAX = 24 #h

EPOCH = datetime(1970, 1, 1)
LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

FIG_W = 960
FIG_H = int(960*LAT_RANGE/LON_RANGE) #533 #768

dict_list = []
for filename in config.l_input_filepath:
    with open(os.path.join(config.dataset_dir,filename),"rb") as f:
        temp = pickle.load(f)
        dict_list.append(temp)


# In[3]:


print(" Remove erroneous timestamps and erroneous speeds...")
Vs = dict()
for Vi,filename in zip(dict_list, config.l_input_filepath):
    print(filename)
    for mmsi in list(Vi.keys()):       
        # Boundary
        lat_idx = np.logical_or((Vi[mmsi][:,LAT] > LAT_MAX),
                                (Vi[mmsi][:,LAT] < LAT_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lat_idx)]
        lon_idx = np.logical_or((Vi[mmsi][:,LON] > LON_MAX),
                                (Vi[mmsi][:,LON] < LON_MIN))
        Vi[mmsi] = Vi[mmsi][np.logical_not(lon_idx)]
#         # Abnormal timestamps
#         abnormal_timestamp_idx = np.logical_or((Vi[mmsi][:,TIMESTAMP] > t_max),
#                                                (Vi[mmsi][:,TIMESTAMP] < t_min))
#         Vi[mmsi] = Vi[mmsi][np.logical_not(abnormal_timestamp_idx)]
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
del dict_list, Vi, abnormal_speed_idx


# In[4]:


print(len(Vs))


# In[5]:


## STEP 2: VOYAGES SPLITTING 
#======================================
# Cutting discontiguous voyages into contiguous ones
print("Cutting discontiguous voyages into contiguous ones...")
count = 0
voyages = dict()
INTERVAL_MAX = 2*3600 # 2h
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


# In[6]:


print(len(Vs))


# In[7]:


# STEP 3: REMOVING SHORT VOYAGES
#======================================
# Removing AIS track whose length is smaller than 20 or those last less than 4h
print("Removing AIS track whose length is smaller than 20 or those last less than 4h...")

for k in list(voyages.keys()):
    duration = voyages[k][-1,TIMESTAMP] - voyages[k][0,TIMESTAMP]
    if (len(voyages[k]) < 20) or (duration < 4*3600):
        voyages.pop(k, None)


# In[8]:


print(len(voyages))


# In[9]:


# STEP 4: REMOVING OUTLIERS
#======================================
print("Removing anomalous message...")
error_count = 0
tick = time.time()
for k in  tqdm(list(voyages.keys())):
    track = voyages[k][:,[TIMESTAMP,LAT,LON,SOG]] # [Timestamp, Lat, Lon, Speed]
    try:
        o_report, o_calcul = utils.detectOutlier(track, speed_max = 30)
        if o_report.all() or o_calcul.all():
            voyages.pop(k, None)
        else:
            voyages[k] = voyages[k][np.invert(o_report)]
            voyages[k] = voyages[k][np.invert(o_calcul)]
    except:
        voyages.pop(k,None)
        error_count += 1
tok = time.time()
print("STEP 4: duration = ",(tok - tick)/60) # 139.685766101 mfrom tqdm import tqdmins


# In[10]:


print(len(voyages))


# In[13]:


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

# In[11]:


## STEP 8: RE-SPLITTING
#======================================
print('Re-Splitting...')
Data = dict()
count = 0
for k in tqdm(list(Vs.keys())): 
    v = Vs[k]
    # Split AIS track into small tracks whose duration <= 1 day
    idx = np.arange(0, len(v), 12*DURATION_MAX)[1:]
    tmp = np.split(v,idx)
    for subtrack in tmp:
        # only use tracks whose duration >= 4 hours
        if len(subtrack) >= 12*4:
            Data[count] = subtrack
            count += 1
print(len(Data))


# ## STEP 5: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
# #======================================
# # Removing 'moored' or 'at anchor' voyages
# print("Removing 'moored' or 'at anchor' voyages...")
# for mmsi in  tqdm(list(voyages.keys())):
#     d_L = float(len(voyages[mmsi]))

#     if np.count_nonzero(voyages[mmsi][:,NAV_STT] == 1)/d_L > 0.7       or np.count_nonzero(voyages[mmsi][:,NAV_STT] == 5)/d_L > 0.7:
#         voyages.pop(mmsi,None)
#         continue
#     sog_max = np.max(voyages[mmsi][:,SOG])
#     if sog_max < 1.0:
#         voyages.pop(mmsi,None)

        
## STEP 5: REMOVING 'MOORED' OR 'AT ANCHOR' VOYAGES
#======================================
# Removing 'moored' or 'at anchor' voyages
print("Removing 'moored' or 'at anchor' voyages...")
for k in  tqdm(list(Data.keys())):
    d_L = float(len(Data[k]))

    if np.count_nonzero(Data[k][:,NAV_STT] == 1)/d_L > 0.7 \
    or np.count_nonzero(Data[k][:,NAV_STT] == 5)/d_L > 0.7:
        Data.pop(k,None)
        continue
    sog_max = np.max(Data[k][:,SOG])
    if sog_max < 1.0:
        Data.pop(k,None)
print(len(Data))
# In[12]:


# In[15]:


## STEP 6: REMOVING LOW SPEED TRACKS
#======================================
print("Removing 'low speed' tracks...")
for k in tqdm(list(Data.keys())):
    d_L = float(len(Data[k]))
    if np.count_nonzero(Data[k][:,SOG] < 2)/d_L > 0.8:
        Data.pop(k,None)
print(len(Data))


# In[21]:


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


# In[22]:


print(config.output_filepath)


# In[23]:


# plt.plot(Data[0][:,LON],Data[0][:,LAT])


# In[24]:


print(len(Data))


# In[25]:


print(os.path.dirname(config.output_filepath))


# In[26]:


os.path.exists(os.path.dirname(config.output_filepath))


# In[27]:


if not os.path.exists(os.path.dirname(config.output_filepath)):
    os.makedirs(os.path.dirname(config.output_filepath))


# In[28]:


## STEP 10: WRITING TO DISK
#======================================
with open(config.output_filepath,"wb") as f:
    pickle.dump(Data,f)


# In[29]:


# print(debug)


# In[30]:


print(len(Data))


# In[31]:


minlen = 1000
for k in list(Data.keys()):
    v = Data[k]
    if len(v) < minlen:
        minlen = len(v)
print("min len: ",minlen)


# In[32]:


# len(Data[0])


# In[33]:


# print(debug)


# In[34]:


## Loading coastline polygon.
# For visualisation purpose, delete this part if you do not have coastline
# shapfile

coastline_filename = "./streetmap_coastline_Bretagne.pkl"

try:
    with open(coastline_filename, 'rb') as f:
        l_coastline_poly = pickle.load(f)
except:
    with open(coastline_filename, 'rb') as f:
        l_coastline_poly = pickle.load(f, encoding='latin1')


# In[35]:


config.output_filepath


# In[36]:


Vs = Data
FIG_DPI = 150
plt.figure(figsize=(FIG_W/FIG_DPI, FIG_H/FIG_DPI), dpi=FIG_DPI)
cmap = plt.cm.get_cmap('Blues')
l_keys = list(Vs.keys())
N = len(Vs)
for d_i in range(N):
    key = l_keys[d_i]
    c = cmap(float(d_i)/(N-1))
    tmp = Vs[key]
    v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
    v_lon = tmp[:,1]*LON_RANGE + LON_MIN
#     plt.plot(v_lon,v_lat,linewidth=0.8)
    plt.plot(v_lon,v_lat,color=c,linewidth=0.8)

## Coastlines
if "bretagne" in config.output_filepath:
    for point in l_coastline_poly:
        poly = np.array(point)
        plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)

plt.xlim([LON_MIN,LON_MAX])
plt.ylim([LAT_MIN,LAT_MAX])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig(config.output_filepath.replace(".pkl",".png"))

