#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:50:23 2018

@author: vnguye04
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

LAT, LON, SOG, COG, HEADING, ROT, NAV_STT, TIMESTAMP, MMSI = list(range(9))

## Training set no cyclone
###############################################################################

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset5/dataset5_train.pkl","rb") as f:
    Vs_train = pickle.load(f)

Vs = Vs_train
for key in list(Vs.keys()):
    tmp = Vs[key]
    plt.plot(tmp[:,1],tmp[:,0])
plt.xlim([0,1])
plt.ylim([0,1])

for key in list(Vs.keys()):
    tmp = Vs[key]
    lat_max = np.max(tmp[:,LAT])
    lon_max = np.max(tmp[:,LON])
    lat_min = np.min(tmp[:,LAT])
    lon_min = np.min(tmp[:,LON])
    if (lat_max < 0.24) and (lat_min > 0.155) and (lon_max<0.53) and (lon_min>0.485):
        Vs.pop(key,None)
    
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/dataset5/dataset5_train_nocyclone.pkl","wb") as f:
    pickle.dump(Vs,f)
    
## Test set moved cyclone
###############################################################################

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_train.pkl","rb") as f:
    Vs_test = pickle.load(f)

Vs = Vs_test
for key in list(Vs.keys()):
    tmp = Vs[key]
    lat_max = np.max(tmp[:,LAT])
    lon_max = np.max(tmp[:,LON])
    lat_min = np.min(tmp[:,LAT])
    lon_min = np.min(tmp[:,LON])
    if (lat_max < 0.24) and (lat_min > 0.155) and (lon_max<0.53) and (lon_min>0.485):
        print(("key: ", key, "mmsi: ", tmp[0,MMSI]))
        Vs[key][:,LON] += 3.8/10.5
        Vs[key][:,LAT] += 0.5/3.5
    else:
        Vs.pop(key,None)

for key in list(Vs.keys()):
    tmp = Vs[key]
    plt.plot(tmp[:,1],tmp[:,0])
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()


with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_test_movedcyclones.pkl","wb") as f:
    pickle.dump(Vs,f)


## Test set only cyclone
###############################################################################

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_train.pkl","rb") as f:
    Vs_test = pickle.load(f)

Vs = Vs_test
for key in list(Vs.keys()):
    tmp = Vs[key]
    lat_max = np.max(tmp[:,LAT])
    lon_max = np.max(tmp[:,LON])
    lat_min = np.min(tmp[:,LAT])
    lon_min = np.min(tmp[:,LON])
    if (lat_max < 0.24) and (lat_min > 0.155) and (lon_max<0.53) and (lon_min>0.485):
        print(("key: ", key, "mmsi: ", tmp[0,MMSI]))
    else:
        Vs.pop(key,None)

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_test_onlycyclones.pkl","wb") as f:
    pickle.dump(Vs,f)

# Step 10: Route Divergence
###############################################################################
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_train.pkl","rb") as f:
    Vs_train = pickle.load(f)
with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_test.pkl","rb") as f:
    Vs_test = pickle.load(f)

#for key in Vs.keys():
#    tmp = Vs[key]
#    lat_begin = tmp[0,LAT]
#    lat_end = tmp[-1,LAT]
#    lon_begin = tmp[0,LON]
#    lon_end = tmp[-1,LON]
#    if (lat_begin < 0.38) and (lat_begin > 0.35) and (lon_begin <0.43) and (lon_begin >0.36):
#        print("key: ", key, "mmsi: ", tmp[0,MMSI])
#    elif (lat_end < 0.38) and (lat_end > 0.35) and (lon_end <0.43) and (lon_end >0.36):
#        print("key: ", key, "mmsi: ", tmp[0,MMSI])        
#    else:
#        Vs.pop(key,None)
        
#for key in Vs.keys():
#    tmp = Vs[key]
#    print(tmp[0,MMSI], len(tmp)/2)
#    plt.figure()
#    plt.plot(tmp[:,1],tmp[:,0])
#    plt.title(str(tmp[0,MMSI]))
#    plt.xlim([0,1])
#    plt.ylim([0,1])

FIG_DPI = 150
plt.figure(figsize=(1920/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)  
cmap = plt.cm.get_cmap('Blues')


Vs = Vs_test
d_i = 0
N = len(Vs)
l_v_true = []
for key in list(Vs.keys()):
    tmp = Vs[key]
    c = cmap(float(d_i)/(N-1))
    d_i+=1
    lat_begin = tmp[0,LAT]
    lat_end = tmp[-1,LAT]
    lon_begin = tmp[0,LON]
    lon_end = tmp[-1,LON]
    if (lat_end < 0.38) and (lat_end > 0.35) and (lon_end <0.43) and (lon_end >0.36):
        if int(tmp[0,MMSI]) == 538200309:
#        if True:
            plt.plot(tmp[:,1],tmp[:,0],color='g',linewidth=2)
            v_true = np.copy(tmp)
            l_v_true.append(v_true)
    else:
        plt.plot(tmp[:,1],tmp[:,0],color=c,linewidth=0.3)
plt.xlim([0,1])
plt.ylim([0,1])

Vs_divergence = dict()
for e in range(-5,6):
    Vs_divergence[e] = np.copy(v_true)
    Vs_divergence[e][:,LAT] += e*0.01
#    Vs_divergence[e][:,LON] += 0.5 # divergences 2 only
    
    
for key in list(Vs_divergence.keys()):
    tmp = Vs_divergence[key]
    plt.plot(tmp[:,1],tmp[:,0],linewidth=0.8)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

with open("/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/MarineC_Jan2014_norm/MarineC_Jan2014_norm_test_divergences.pkl","wb") as f:
    pickle.dump(Vs_divergence,f)

