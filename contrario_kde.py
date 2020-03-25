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
A contrario detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import glob
from scipy import stats
from tqdm import tqdm
import contrario_utils

## PARAMS
#======================================
# Bretagne dataset
LAT_MIN = 47.5
LAT_MAX = 49.5
LON_MIN = -7.0
LON_MAX = -4.0
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
FIG_DPI = 150
LAT_RESO = 0.1
LON_RESO = LAT_RESO
LAT_BIN = int(LAT_RANGE/LAT_RESO)
LON_BIN = int(LON_RANGE/LON_RESO)
LATENT_SIZE = 100
CONTRARIO_EPS = 1e-9
MISSING_DATA = False

ONEHOT_LON_BINS = 300
ONEHOT_LAT_BINS = 200

print("EPSILON ",CONTRARIO_EPS)

trainingset_name ="ct_2017010203_10_20"
testset_name ="ct_2017010203_10_20"



# trained step
step = 80002

# save_dir = "./results/"+trainingset_name+"/contrario_"+str(LATENT_SIZE)+"_"+str(step)+"/"
save_dir = "./results/"+trainingset_name+"/log_density-"+trainingset_name+"_train.pkl-"\
+trainingset_name+"_valid.pkl-"+str(LATENT_SIZE)+"-missing_data-"+str(MISSING_DATA)+"-step-"+str(step)+"/"



## LOADING SAVED DATA
#======================================

## Loading coastline polygon.
# For visualisation purpose, delete this part if you do not have the coastline
# shapfile

try:
    with open("./data/"+trainingset_name+"/"+trainingset_name+"_train.pkl","rb") as f:
        Vs_train = pickle.load(f)
    with open("./data/"+trainingset_name+"/"+trainingset_name+"_valid.pkl","rb") as f:
        Vs_valid = pickle.load(f)
    with open("./data/"+testset_name+"/"+testset_name+"_test.pkl","rb") as f:
        Vs_test = pickle.load(f)
except:
    with open("./data/"+trainingset_name+"/"+trainingset_name+"_train.pkl","rb") as f:
        Vs_train = pickle.load(f, encoding='latin1')
    with open("./data/"+trainingset_name+"/"+trainingset_name+"_valid.pkl","rb") as f:
        Vs_valid = pickle.load(f, encoding='latin1')
    with open("./data/"+testset_name+"/"+testset_name+"_test.pkl","rb") as f:
        Vs_test = pickle.load(f, encoding='latin1')




## LOADING SAVED LOG_DENSITY MAP
#======================================
## Loading the parameters of the distribution in each cell (calculated by the
# tracks in the validation set)
m_map_ll_mean = np.load(save_dir+"map_ll_mean-"+str(LAT_RESO)+"-"+str(LON_RESO) + ".npy")
m_map_ll_std = np.load(save_dir+"map_ll_std-"+str(LAT_RESO)+"-"+str(LON_RESO) + ".npy")
with open(save_dir+"map_ll-"+str(LAT_RESO)+"-"+str(LON_RESO)+".pkl","rb") as f:
    Map_ll = pickle.load(f)


## Loading the log[p(x_t|h_t)] of AIS tracks in the test set
save_filename = "outcomes-"+trainingset_name+"_train.pkl-"\
                +testset_name+"_test.pkl"\
                +"-" + str(LATENT_SIZE)\
                +"-missing_data-"+str(MISSING_DATA)\
                +"-step-"+str(step)\
                +".pkl"

try:
    with open("./results/"+trainingset_name+"/"+save_filename,"rb") as f:
        l_dict = pickle.load(f)
except:
    with open("./results/"+trainingset_name+"/"+save_filename,"rb") as f:
        l_dict = pickle.load(f, encoding='latin1')
        
        
## PROCESSING THE "A CONTRARIO" ANOMALY DETECTION
#======================================

d_i = 0
v_mean_log = []
l_v_A = []
v_buffer_count = []
length_track = len(l_dict[0]["inp"])
l_dict_anomaly = []
for D in tqdm(l_dict):
    tmp = D["inp"]
    m_log_weights_np = D["log_weights"]
    v_A = np.zeros(len(tmp))
    for d_timestep in range(2*6,len(tmp)):
        d_row = int(tmp[d_timestep,0]*0.01/LAT_RESO)
        d_col = int((tmp[d_timestep,1]-ONEHOT_LAT_BINS)*0.01/LON_RESO)
        d_ll_t = np.mean(m_log_weights_np[d_timestep,:])
        
        ## KDE
        # Use KDE to estimate the distribution of log[p(x_t|h_t)] in each cell.
        l_local_log_prod = Map_ll[str(d_row)+","+str(d_col)]
        if len(l_local_log_prod) < 2: 
            # Ignore cells that do not have enough data.
            v_A[d_timestep] = 2
        else:
            kernel = stats.gaussian_kde(l_local_log_prod)
            cdf = kernel.integrate_box_1d(-np.inf,d_ll_t)
            if cdf < 0.1:
                v_A[d_timestep] = 1
    # log[p(x_t|h_t)] of the first timesteps of the tracks may not robust, 
    # because h_t was initialized as a zeros.
    v_A = v_A[12:]
    v_anomalies = np.zeros(len(v_A))
    for d_i_4h in range(0,len(v_A)+1-24):
        v_A_4h = v_A[d_i_4h:d_i_4h+24]
        v_anomalies_i = contrario_utils.contrario_detection(v_A_4h,CONTRARIO_EPS)
        v_anomalies[d_i_4h:d_i_4h+24][v_anomalies_i==1] = 1

    ## Plotting
#    tmp = tmp[12:]
#    v_lat = (tmp[:,0]/float(ONEHOT_LAT_BINS))*LAT_RANGE + LAT_MIN
#    v_lon = ((tmp[:,1]-ONEHOT_LAT_BINS)/float(ONEHOT_LON_BINS))*LON_RANGE + LON_MIN
#    plt.plot(v_lon,v_lat,color='g',linewidth=0.8)
#    for l_segment in nonzero_segments(v_anomalies):
#        plt.plot(v_lon[l_segment],v_lat[l_segment],color='r',linewidth=0.8)
    if len(contrario_utils.nonzero_segments(v_anomalies)) > 0:
        D["anomaly_idx"] = v_anomalies
        l_dict_anomaly.append(D)

print("Number of abnormal tracks: ",len(l_dict_anomaly))

## SAVING TO DISK
#======================================
d_n_anomalies = len(l_dict_anomaly)
print("Number of abnormal tracks detected: ",len(l_dict_anomaly))
save_pkl_filename = save_dir\
                    +save_filename.replace("outcomes","kde")\
                                        +"-resolution-"+str(LAT_RESO)\
                                        +"-epsilon-"+str(CONTRARIO_EPS)\
                                        +"-missing_data-" + str(MISSING_DATA)\
                                        +"-"+str(d_n_anomalies)+".pkl"
with open(save_pkl_filename,"wb") as f:
    pickle.dump(l_dict_anomaly,f)
    
    
## PLOTTING ABNORMAL TRACKS
#======================================
"""
VISUALISING
"""
d_n_anomalies = len(l_dict_anomaly)

plt.figure(figsize=(960/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)

## Plot Vs_train (blue)
Vs = Vs_train
cmap = plt.cm.get_cmap('Blues')
l_keys = list(Vs.keys())
N = len(Vs)
for d_i in range(N):
    key = l_keys[d_i]
    c = cmap(float(d_i)/(N-1))
    tmp = Vs[key]
    v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
    v_lon = tmp[:,1]*LON_RANGE + LON_MIN
    plt.plot(v_lon,v_lat,color=c,linewidth=0.8)
plt.xlim([LON_MIN,LON_MAX])
plt.ylim([LAT_MIN,LAT_MAX])
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()


# ## Plot Vs_test (green)
# Vs = Vs_test
# cmap = plt.cm.get_cmap('Greens')
# l_keys = list(Vs.keys())
# N = len(Vs)
# for d_i in range(N):
#     key = l_keys[d_i]
#     c = cmap(float(d_i)/(N-1))
#     tmp = Vs[key]
#     v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
#     v_lon = tmp[:,1]*LON_RANGE + LON_MIN
#     plt.plot(v_lon,v_lat,color=c,linewidth=0.5)
# plt.xlim([LON_MIN,LON_MAX])
# plt.ylim([LAT_MIN,LAT_MAX])
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.tight_layout()

## Coastlines
## Loading coastline polygon.
# For visualisation purpose, delete this part if you do not have the coastline
# shapfile

cmap_anomaly = plt.cm.get_cmap('autumn')
N_anomaly = len(l_dict_anomaly)
d_i = 0
for D in l_dict_anomaly:
    try:
        c = cmap_anomaly(float(d_i)/(N_anomaly-1))
    except:
        c = 'r'
    d_i += 1
    tmp = D["inp"]
    m_log_weights_np = D["log_weights"]
    tmp = tmp[12:]
    v_lat = (tmp[:,0]/float(ONEHOT_LAT_BINS))*LAT_RANGE + LAT_MIN
    v_lon = ((tmp[:,1]-ONEHOT_LAT_BINS)/float(ONEHOT_LON_BINS))*LON_RANGE + LON_MIN
    plt.plot(v_lon,v_lat,color=c,linewidth=1.2)


fig_name = save_dir\
            +save_filename.replace("outcomes","kde")\
                                        +"-resolution-"+str(LAT_RESO)\
                                        +"-epsilon-"+str(CONTRARIO_EPS)\
                                        +"-missing_data-" + str(MISSING_DATA)\
                                        +"-"+str(d_n_anomalies)+".png"
plt.savefig(fig_name,dpi = FIG_DPI)
