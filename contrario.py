"""
Implementation of a simple and naive "a contrario" anomaly detectio model for 
MultitaskAIS 
https://arxiv.org/abs/1806.03972

For the "a contrario" model, please refer to:
A. Desolneux, L. Moisan, and J.-M. Morel, From Gestalt Theory to Image Analysis, 
vol. 34. New York, NY: Springer New York, 2008.

We divide the ROI into small cells and suppose the log[p(x_t|x_{1..t-1},z_{1..t-1})]
in each cell is normally distributed (naive hypothesis). The mean and the variance
of the distribution in each cell are calculated emperitially by the track in the
validation set

A point in a AIS track will be considered as abnormal point if 
            log[p(x_t|x_{1..t-1},z_{1..t-1})] < mu - z_p*sigma
Where:
    mu      : the mean of the log[p] in this cell
    sigma   : the std of the log[p] in this cell
    z_p     : the quantile (hyperparameter)

A segment X = {x_t1..x_t2} of a AIS track will be considered as abnormal depends
on the number of abnormal points in this segment and its length.
"""
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import contrario_utils

# Bretagne dataset
LAT_MIN = 47.0
LAT_MAX = 50.0
LON_MIN = -7.0
LON_MAX = -4.0
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
FIG_DPI = 300
LAT_RESO = 0.1
LON_RESO = LAT_RESO
LAT_BIN = int(LAT_RANGE/LAT_RESO)
LON_BIN = int(LON_RANGE/LON_RESO)
CONTRARIO_EPS = 1e-9
MISSING_DATA = False

print("LAT_RESO",LAT_RESO,"EPSILON ",CONTRARIO_EPS)



# ==============================================================================
"""
LOADING SAVED DATA
"""
## Loading coastline polygon. 
# For visualisation purpose, delete this part if you do not have coastline 
# shapfile
"""
coastline_filename = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/"\
                     + "coastlines-split-4326/streetmap_coastline_Bretagne.pkl"
with open(coastline_filename, 'r') as f: 
    l_coastline_poly = pickle.load(f)
"""

## Loading AIS tracks in the training set. 
try:
    with open("./data/dataset8/dataset8_train.pkl","rb") as f:
        Vs_train = pickle.load(f)
except:
    with open("./data/dataset8/dataset8_train.pkl","rb") as f:
        Vs_train = pickle.load(f, encoding='latin1')

## Plotting AIS tracks inthe training set
Vs = Vs_train
plt.figure(figsize=(960*2/FIG_DPI, 960*2/FIG_DPI), dpi=FIG_DPI)  
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


## Loading the parameters of the distribution in each cell (calculated by the
# tracks in the validation set) 
save_dir = "./results/dataset8/log_density-dataset8_train.pkl-dataset8_valid.pkl-100-missing_data-"+str(MISSING_DATA)+"/"
m_map_ll_mean = np.load(save_dir+"map_ll_mean-"+str(LAT_RESO)+"-"+str(LON_RESO) + ".npy")
m_map_ll_std = np.load(save_dir+"map_ll_std-"+str(LAT_RESO)+"-"+str(LON_RESO) + ".npy")


## Loading the log[p(x_t|x_{1..t-1},z_{1..t-1})] of points in AIS tracks in 
# the test set
save_dir = "./results/dataset8/"
save_filename = "outcomes-dataset8_train.pkl-dataset8_test.pkl-100-missing_data-"+str(MISSING_DATA)+".pkl"

try:
    with open(save_dir+save_filename,"rb") as f:
        l_dict = pickle.load(f)
except:
    with open(save_dir+save_filename,"rb") as f:
        l_dict = pickle.load(f, encoding='latin1')


# ==============================================================================
"""
PROCESSING THE "A CONTRARIO" ANOMALY DETECTION
"""
d_i = 0
v_mean_log = []
l_v_A = []
v_buffer_count = []
length_track = len(l_dict[0]["inp"])
count = 0
l_dict_anomaly = []
for D in l_dict:
    count += 1
    print(count)
    tmp = D["inp"]
    m_log_weights_np = D["log_weights"]
    v_A = np.zeros(len(tmp))
    for d_timestep in range(2*6,len(tmp)):
        d_row = int(tmp[d_timestep,0]*0.01/LAT_RESO)
        d_col = int((tmp[d_timestep,1]-300)*0.01/LON_RESO)
        d_ll_t = np.mean(m_log_weights_np[d_timestep,:,:])  
        d_qualtile = (d_ll_t - m_map_ll_mean[d_row,d_col])/m_map_ll_std[d_row,d_col]
        if (not np.isfinite(d_qualtile)):
            v_A[d_timestep] = 2
        elif (d_qualtile <= -1.282):
            v_A[d_timestep] = 1
    v_A = v_A[12:]
    v_anomalies = np.zeros(len(v_A))
    for d_i_4h in range(0,len(v_A)+1-24):
        v_A_4h = v_A[d_i_4h:d_i_4h+24]
        v_anomalies_i = contrario_utils.contrario_detection(v_A_4h,CONTRARIO_EPS)
        v_anomalies[d_i_4h:d_i_4h+24][v_anomalies_i==1] = 1
    
    ## Plotting
    tmp = tmp[12:]
    v_lat = (tmp[:,0]/300.0)*LAT_RANGE + LAT_MIN
    v_lon = (tmp[:,1]-300.0)/300.0*LON_RANGE + LON_MIN
    plt.plot(v_lon,v_lat,color='g',linewidth=0.8)
    if len(contrario_utils.nonzero_segments(v_anomalies)) > 0:
        D["anomaly_idx"] = v_anomalies
        l_dict_anomaly.append(D)
#    for l_segment in nonzero_segments(v_anomalies):
#        plt.plot(v_lon[l_segment],v_lat[l_segment],color='r',linewidth=0.8)
    ## 
#plt.close()


# ==============================================================================-True
"""
VISUALISING
"""

## Coastlines
"""
for point in l_coastline_poly:
    poly = np.array(point)
    plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)
"""
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
    v_lat = (tmp[:,0]/300.0)*LAT_RANGE + LAT_MIN
    v_lon = (tmp[:,1]-300.0)/300.0*LON_RANGE + LON_MIN
    plt.plot(v_lon,v_lat,color=c,linewidth=1.2)

## Abnormal tracks
d_n_anomalies = len(l_dict_anomaly)
print("Number of abnormal tracks detected: ",len(l_dict_anomaly))

if not os.path.exists(save_dir+"contrario/"):
    os.makedirs(save_dir+"contrario/")

save_pkl_filename = save_dir+"contrario/"\
                    +save_filename.replace("outcomes","contrario")\
                                        +"-resolution-"+str(LAT_RESO)\
                                        +"-epsilon-"+str(CONTRARIO_EPS)\
                                        +"-missing_data-" + str(MISSING_DATA)\
                                        +"-"+str(d_n_anomalies)+".pkl"
with open(save_pkl_filename,"wb") as f:
    pickle.dump(l_dict_anomaly,f)

fig_name = save_dir+"contrario/"\
            +save_filename.replace("outcomes","contrario")\
                                        +"-resolution-"+str(LAT_RESO)\
                                        +"-epsilon-"+str(CONTRARIO_EPS)\
                                        +"-missing_data-" + str(MISSING_DATA)\
                                        +"-"+str(d_n_anomalies)+".png"
plt.savefig(fig_name,dpi = FIG_DPI)
plt.close()


Vs = Vs_train
l_keys = list(Vs.keys())
N = len(Vs)
d_i_anomaly = 0
cmap_anomaly = plt.cm.get_cmap('autumn')
N_anomaly = len(l_dict_anomaly)
for D in l_dict_anomaly:
    plt.figure(figsize=(960*2/FIG_DPI, 960*2/FIG_DPI), dpi=FIG_DPI)  
    # Trajets in training set
    cmap = plt.cm.get_cmap('Blues')
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
    
    # Coastlines
    for point in l_coastline_poly:
        poly = np.array(point)
        plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)
    
    # Abnormal track
    try:
        c = cmap_anomaly(float(d_i_anomaly)/(N_anomaly-1))
    except:
        c = 'r'
    d_i_anomaly += 1
    tmp = D["inp"]
    m_log_weights_np = D["log_weights"]
    tmp = tmp[12:]
    v_anomalies = D["anomaly_idx"]
    v_lat = (tmp[:,0]/300.0)*LAT_RANGE + LAT_MIN
    v_lon = (tmp[:,1]-300.0)/300.0*LON_RANGE + LON_MIN
    plt.plot(v_lon, v_lat, color=c,linewidth=1.5)
#    plt.plot(v_lon, v_lat, color=c,linewidth=1.5,marker='o')
#    plt.plot(v_lon[0],v_lat[0],color=c,linewidth=1.2,marker="x")
#    for l_segment in contrario_utils.nonzero_segments(v_anomalies):
#        plt.plot(v_lon[l_segment],v_lat[l_segment],color=c,linewidth=1.2,linestyle="--")
#    for l_segment in contrario_utils.zero_segments(v_anomalies):
#        plt.plot(v_lon[l_segment],v_lat[l_segment],color=c,linewidth=1.2,linestyle="-")   
    fig_name = save_dir+"contrario/"\
            +save_filename.replace("outcomes","contrario")+\
                                        "-resolution-"+str(LAT_RESO)+\
                                        "-epsilon-"+str(CONTRARIO_EPS)+"-"+str(d_n_anomalies)\
                                        +"-"+str(d_i_anomaly)+".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()
    

