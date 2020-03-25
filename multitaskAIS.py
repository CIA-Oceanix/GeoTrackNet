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
A script to run the task-specific blocks of MultitaskAIS.
The code is adapted from
https://github.com/tensorflow/models/tree/master/research/fivo


    USAGE:
    python multitaskAIS.py \
    --mode=train \
    --logdir=./chkpt \
    --bound=elbo \
    --summarize_every=100 \
    --latent_size=100 \
    --batch_size=50 \
    --num_samples=16 \
    --learning_rate=0.0003 \

"""

import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.ndimage as ndimage
import pickle
from tqdm import tqdm
import logging
import math
import scipy.special
from scipy import stats
from tqdm import tqdm
import csv
from datetime import datetime
import utils
import contrario_utils


import runners
from flags_config import config

LAT_RANGE = config.lat_max - config.lat_min
LON_RANGE = config.lon_max - config.lon_min

FIG_DPI = 150
FIG_W = 960
FIG_H = int(FIG_W*LAT_RANGE/LON_RANGE)

LOGPROB_MEAN_MIN = -10.0
LOGPROB_STD_MAX = 5

## RUN TRAIN
#======================================

if config.mode == "train":
    print(config.trainingset_path)
    fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
    tf.logging.set_verbosity(tf.logging.INFO)
    # get TF logger
    logger = logging.getLogger('tensorflow')
    logger.addHandler(fh)
    runners.run_train(config)

else:
    with open(config.testset_path,"rb") as f:
        Vs_test = pickle.load(f)
    dataset_size = len(Vs_test)

## RUN TASK-SPECIFIC SUBMODEL
#======================================
  
step = None
if config.mode in ["save_logprob","traj_reconstruction"]:
    tf.Graph().as_default()
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, mmsis, time_starts, time_ends, lengths, model = runners.create_dataset_and_model(config,
                                                               shuffle=False,
                                                               repeat=False)

    if config.mode == "traj_reconstruction":
        config.missing_data = True
    #else:
    #    config.missing_data = False

    track_sample, track_true, log_weights, ll_per_t, ll_acc,_,_,_\
                                        = runners.create_eval_graph(inputs, targets,
                                                               lengths, model, config)
    saver = tf.train.Saver()
    sess = tf.train.SingularMonitoredSession()
    runners.wait_for_checkpoint(saver, sess, config.logdir)
    step = sess.run(global_step)

#runners.wait_for_checkpoint(saver, sess, config.logdir)
#step = sess.run(global_step)
#print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

if step is None:
    # The log filename contains the step.
    index_filename = sorted(glob.glob(config.logdir+"/*.index"))[-1] # the lastest step
    step = int(index_filename.split(".index")[0].split("ckpt-")[-1])
    

print("Global step: ", step)
outputs_path = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "logprob-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name) + "-"\
            + str(config.latent_size)\
            + "-missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            + ".pkl"
if not os.path.exists(os.path.dirname(outputs_path)):
    os.makedirs(os.path.dirname(outputs_path))

save_dir = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "local_logprob-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name).replace("test","valid") + "-"\
            + str(config.latent_size) + "-"\
            + "missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            +"/"      

#===============================================================================
#===============================================================================
if config.mode == "save_logprob":
    """ save_logprob
    Calculate and save log[p(x_t|h_t)] of each track in the test set.
    """
    l_dict = []
    for d_i in tqdm(list(range(math.ceil(dataset_size/config.batch_size)))):
        inp, tar, mmsi, t_start, t_end, seq_len, log_weights_np, true_np, ll_t =\
                 sess.run([inputs, targets, mmsis, time_starts, time_ends, lengths, log_weights, track_true, ll_per_t])
        for d_idx_inbatch in range(inp.shape[1]):
            D = dict()
            seq_len_d = seq_len[d_idx_inbatch]
            D["seq"] = np.nonzero(tar[:seq_len_d,d_idx_inbatch,:])[1].reshape(-1,4)
            D["t_start"] = t_start[d_idx_inbatch]
            D["t_end"] = t_end[d_idx_inbatch]
            D["mmsi"] = mmsi[d_idx_inbatch]
            D["log_weights"] = log_weights_np[:seq_len_d,:,d_idx_inbatch]
            l_dict.append(D)
    with open(outputs_path,"wb") as f:
        pickle.dump(l_dict,f)

    """ LL
    Plot the distribution of log[p(x_t|h_t)] of each track in the test set.
    """

    v_logprob = np.empty((0,))
    v_logprob_stable = np.empty((0,))

    count = 0
    for D in tqdm(l_dict):
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        v_logprob = np.concatenate((v_logprob,[ll_t]))

    d_mean = np.mean(v_logprob)
    d_std = np.std(v_logprob)
    d_thresh = d_mean - 3*d_std

    plt.figure(figsize=(1920/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)
    plt.plot(v_logprob,'o')
    plt.title("Log likelihood " + os.path.basename(config.testset_name)\
              + ", mean = {0:02f}, std = {1:02f}, threshold = {2:02f}".format(d_mean, d_std, d_thresh))
    plt.plot([0,len(v_logprob)], [d_thresh, d_thresh],'r')

    plt.xlim([0,len(v_logprob)])
    fig_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/" \
            + "logprob-" \
            + config.bound + "-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name)\
            + "-latent_size-" + str(config.latent_size)\
            + "-ll_thresh" + str(round(d_thresh, 2))\
            + "-missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            + ".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()
    
#===============================================================================
#===============================================================================
elif config.mode == "local_logprob":
    """ LOCAL THRESHOLD
    The ROI is divided into small cells, in each cell, we calculate the mean and
    the std of log[p(x_t|h_t)].
    """
    # Init
    m_map_logprob_std = np.zeros(shape=(config.n_lat_cells,config.n_lon_cells))
    m_map_logprob_mean = np.zeros(shape=(config.n_lat_cells,config.n_lon_cells))
    m_map_density = np.zeros(shape=(config.n_lat_cells,config.n_lon_cells))
    v_logprob = np.empty((0,))
    v_mmsi = np.empty((0,))
    Map_logprob = dict()
    for row  in range(config.n_lat_cells):
        for col in range(config.n_lon_cells):
            Map_logprob[ str(str(row)+","+str(col))] = []
    
    # Load logprob
    with open(outputs_path,"rb") as f:
        l_dict = pickle.load(f)

    print("Calculating the logprob map...")
    for D in tqdm(l_dict):
        tmp = D["seq"]
        log_weights_np = D["log_weights"]
        for d_timestep in range(2*6,len(tmp)):
            row = int(tmp[d_timestep,0]*0.01/config.cell_lat_reso)
            col = int((tmp[d_timestep,1]-config.onehot_lat_bins)*0.01/config.cell_lat_reso)
            Map_logprob[str(row)+","+str(col)].append(np.mean(log_weights_np[d_timestep,:]))

    # Remove outliers
    for row  in range(config.n_lat_cells):
        for col in range(config.n_lon_cells):
            s_key = str(row)+","+str(col) 
            Map_logprob[s_key] = utils.remove_gaussian_outlier(np.array(Map_logprob[s_key]))
            m_map_logprob_mean[row,col] = np.mean(Map_logprob[s_key])
            m_map_logprob_std[row,col] = np.std(Map_logprob[s_key])
            m_map_density[row,col] = len(Map_logprob[s_key])
    
    # Save to disk
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+"map_density-"+str(config.cell_lat_reso)+"-"+str(config.cell_lon_reso),m_map_density)
    with open(os.path.join(save_dir,"Map_logprob-"+str(config.cell_lat_reso)+"-"+str(config.cell_lon_reso)+".pkl"),"wb") as f:
        pickle.dump(Map_logprob,f)
    
    # Show the map
    utils.show_logprob_map(m_map_logprob_mean, m_map_logprob_std, save_dir, 
                           logprob_mean_min = LOGPROB_MEAN_MIN,
                           logprob_std_max = LOGPROB_STD_MAX,
                           fig_w = FIG_W, fig_h = FIG_H,
                          )    

#===============================================================================
#===============================================================================
elif config.mode == "contrario_detection":
    """ CONTRARIO DETECTION
    Detect abnormal vessels' behavior using a contrario detection.
    An AIS message is considered as abnormal if it does not follow the learned 
    distribution. An AIS track is considered as abnormal if many of its messages
    are abnormal.
    """      
    
    # Loading the parameters of the distribution in each cell (calculated by the
    # tracks in the validation set)
    with open(os.path.join(save_dir,"Map_logprob-"+\
              str(config.cell_lat_reso)+"-"+str(config.cell_lat_reso)+".pkl"),"rb") as f:
        Map_logprob = pickle.load(f)
    # Load the logprob
    with open(outputs_path,"rb") as f:
        l_dict = pickle.load(f)
    d_i = 0
    v_mean_log = []
    l_v_A = []
    v_buffer_count = []
    length_track = len(l_dict[0]["seq"])
    l_dict_anomaly = []
    n_error = 0
    for D in tqdm(l_dict):
        try:
        # if True:
            tmp = D["seq"]
            m_log_weights_np = D["log_weights"]
            v_A = np.zeros(len(tmp))
            for d_timestep in range(2*6,len(tmp)):
                d_row = int(tmp[d_timestep,0]*config.onehot_lat_reso/config.cell_lat_reso)
                d_col = int((tmp[d_timestep,1]-config.onehot_lat_bins)*config.onehot_lat_reso/config.cell_lon_reso)
                d_logprob_t = np.mean(m_log_weights_np[d_timestep,:])

                # KDE
                l_local_log_prod = Map_logprob[str(d_row)+","+str(d_col)]
                if len(l_local_log_prod) < 2:
                    v_A[d_timestep] = 2
                else:
                    kernel = stats.gaussian_kde(l_local_log_prod)
                    cdf = kernel.integrate_box_1d(-np.inf,d_logprob_t)
                    if cdf < 0.1:
                        v_A[d_timestep] = 1
            v_A = v_A[12:]
            v_anomalies = np.zeros(len(v_A))
            for d_i_4h in range(0,len(v_A)+1-24):
                v_A_4h = v_A[d_i_4h:d_i_4h+24]
                v_anomalies_i = contrario_utils.contrario_detection(v_A_4h,config.contrario_eps)
                v_anomalies[d_i_4h:d_i_4h+24][v_anomalies_i==1] = 1

            if len(contrario_utils.nonzero_segments(v_anomalies)) > 0:
                D["anomaly_idx"] = v_anomalies
                l_dict_anomaly.append(D)
        except:
            n_error += 1
    print("Number of processed tracks: ",len(l_dict))
    print("Number of abnormal tracks: ",len(l_dict_anomaly)) 
    print("Number of errors: ",n_error)
    
    # Save to disk
    n_anomalies = len(l_dict_anomaly)
    save_filename = os.path.basename(config.trainingset_name)\
                    +"-" + os.path.basename(config.trainingset_name)\
                    +"-" + str(config.latent_size)\
                    +"-missing_data-"+str(config.missing_data)\
                    +"-step-"+str(step)\
                    +".pkl"
    save_pkl_filename = os.path.join(save_dir,"List_abnormal_tracks-"+save_filename)
    with open(save_pkl_filename,"wb") as f:
        pickle.dump(l_dict_anomaly,f)
    
    ## Plot
    with open(config.trainingset_path,"rb") as f:
        Vs_train = pickle.load(f)
    with open(config.testset_path,"rb") as f:
        Vs_test = pickle.load(f)

    save_filename = "Abnormal_tracks"\
                + "-" + os.path.basename(config.trainingset_name)\
                + "-" + os.path.basename(config.testset_name)\
                + "-latent_size-" + str(config.latent_size)\
                + "-step-"+str(step)\
                + "-eps-"+str(config.contrario_eps)\
                + "-" + str(n_anomalies)\
                + ".png"
    
    # Plot abnormal tracks with the tracks in the training set as the background
    utils.plot_abnormal_tracks(Vs_train,l_dict_anomaly,
                     os.path.join(save_dir,save_filename),
                     config.lat_min,config.lat_max,config.lon_min,config.lon_max,
                     config.onehot_lat_bins,config.onehot_lon_bins,
                     background_cmap = "Blues",
                     fig_w = FIG_W, fig_h = FIG_H,
                    )
    plt.close()
    # Plot abnormal tracks with the tracks in the test set as the background
    utils.plot_abnormal_tracks(Vs_test,l_dict_anomaly,
                     os.path.join(save_dir,save_filename.replace("Abnormal_tracks","Abnormal_tracks2")),
                     config.lat_min,config.lat_max,config.lon_min,config.lon_max,
                     config.onehot_lat_bins,config.onehot_lon_bins,
                     background_cmap = "Greens",
                     fig_w = FIG_W, fig_h = FIG_H,
                    )
    plt.close()   
    # Save abnormal tracks to csv file
    with open(os.path.join(save_dir,save_filename.replace(".png",".csv")),"w") as f:
        writer = csv.writer(f)
        writer.writerow(["MMSI","Time_start","Time_end","Timestamp_start","Timestamp_end"])
        for D in l_dict_anomaly:
            writer.writerow([D["mmsi"], 
                             datetime.utcfromtimestamp(D["t_start"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                             datetime.utcfromtimestamp(D["t_end"]).strftime('%Y-%m-%d %H:%M:%SZ'),
                             D["t_start"],D["t_end"]])
#===============================================================================
#===============================================================================
elif config.mode == "p_density":
    """ P DENSITY
    Calculate the mean and std map of p(x_t|h_t)
    We divide the ROI into small cells, in each cell, we calculate the mean and
    the std of p(x_t|h_t).
    """
    Map_logprob = dict()
    for row  in range(config.n_lat_cells):
        for col in range(config.n_lon_cells):
            Map_logprob[ str(str(row)+","+str(col))] = []
    m_map_logprob_std = np.zeros(shape=(config.n_lat_cells,config.n_lon_cells))
    m_map_logprob_mean = np.zeros(shape=(config.n_lat_cells,config.n_lon_cells))
    m_map_density = np.zeros(shape=(config.n_lat_cells,config.n_lon_cells))
    v_logprob = np.empty((0,))
    v_mmsi = np.empty((0,))

    try:
        with open(outputs_path,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outputs_path,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")

    print("Calculatint ll map...")
    for D in tqdm(l_dict):
        tmp = D["seq"]
        log_weights_np = D["log_weights"]
        weights_np = np.exp(scipy.special.logsumexp(log_weights_np,axis=1))
        for d_timestep in range(2*6,len(tmp)):
            row = int(tmp[d_timestep,0]*0.01/config.cell_lat_reso)
            col = int((tmp[d_timestep,1]-config.onehot_lat_bins)*0.01/config.cell_lon_reso)
            Map_logprob[str(row)+","+str(col)].append(np.mean(weights_np[d_timestep,:]))

    def remove_gaussian_outlier(v_data,quantile=1.64):
        d_mean = np.mean(v_data)
        d_std = np.std(v_data)
        idx_normal = np.where(np.abs(v_data-d_mean)<=quantile*d_std)[0] #90%
        return v_data[idx_normal]

    for row  in range(config.n_lat_cells):
        for col in range(config.n_lon_cells):
            v_cell = np.copy(Map_logprob[str(row)+","+str(col)])
#            if len(v_cell) >1 and len(v_cell) < 5:
#                break
            v_cell = remove_gaussian_outlier(v_cell)
            m_map_logprob_mean[row,col] = np.mean(v_cell)
            m_map_logprob_std[row,col] = np.std(v_cell)
            m_map_density[row,col] = len(v_cell)

    save_dir = "results/"\
                + config.trainingset_path.split("/")[-2] + "/"\
                + "p_density-"\
                + os.path.basename(config.trainingset_name) + "-"\
                + os.path.basename(config.testset_name) + "-"\
                + str(config.latent_size) + "-"\
                + "missing_data-" + str(config.missing_data)\
                + "-step-"+str(step)\
                +"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+"map_density-"+str(config.cell_lat_reso)+"-"+str(config.cell_lon_reso),m_map_density)
    np.save(save_dir+"map_logprob_mean-"+str(config.cell_lat_reso)+"-"+str(config.cell_lon_reso),m_map_logprob_mean)
    np.save(save_dir+"map_logprob_std-"+str(config.cell_lat_reso)+"-"+str(config.cell_lon_reso),m_map_logprob_std)

    with open(os.path.join(save_dir,"map_logprob-"+str(config.cell_lat_reso)+"-"+str(config.cell_lon_reso)+".pkl"),"wb") as f:
        pickle.dump(Map_logprob,f)


#===============================================================================
#===============================================================================
elif config.mode == "visualisation":
    """ VISUALISATION
    Visualize the outcome of the global thresholding detection.
    Tracks in the training set will be displayed in blue, normal tracks in the
    test set will be displayed in green, while abnormal tracks in the test set
    will be displayed in red.
    """
    # Plot trajectories in the training set
    try:
        with open(config.trainingset_path,"rb") as f:
            Vs_train = pickle.load(f)
        with open(config.testset_path,"rb") as f:
           Vs_test = pickle.load(f)
    except:
        with open(config.trainingset_path,"rb") as f:
            Vs_train = pickle.load(f,encoding="latin1")
        with open(config.testset_path,"rb") as f:
           Vs_test = pickle.load(f,encoding="latin1")


    print("Plotting tracks in the training set...")
    plt.figure(figsize=(960/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)
    cmap = plt.cm.get_cmap('Blues')
    l_keys = list(Vs_train.keys())
    N = len(Vs_train)
    for d_i in tqdm(list(range(N))):
        key = l_keys[d_i]
        c = cmap(float(d_i)/(N-1))
        tmp = Vs_train[key]
        v_lat = tmp[:,0]*LAT_RANGE + config.lat_min
        v_lon = tmp[:,1]*LON_RANGE + config.lon_min
        plt.plot(v_lon,v_lat,color=c,linewidth=0.8)
#        plt.plot(v_lon,v_lat,color='b',linewidth=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Load the outcomes of the embedding layer
    try:
        with open(outputs_path,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outputs_path,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")

    v_logprob = np.empty((0,))
    v_mmsi = np.empty((0,))
#    print("Plotting tracks in the test set...")
#    for D in tqdm(l_dict):
#        m_tar = D["seq"]
#        log_weights_np = D["log_weights"]
#        ll_t = np.mean(log_weights_np)
#        if len(m_tar) < config.min_duration:
#            continue
#        v_lat = (m_tar[:,0]/float(config.onehot_lat_bins))*LAT_RANGE + config.lat_min
#        v_lon = (m_tar[:,1]-float(config.onehot_lat_bins))/config.onehot_lon_bins*LON_RANGE + config.lon_min
#        v_logprob = np.concatenate((v_logprob,[ll_t]))
#        ll_stable = np.array([np.mean(log_weights_np[2*6,:])])
#        if config.mode == "superposition_stable":
#            ll_track = ll_stable
#        else:
#            ll_track = ll_t
#        if ll_track >= config.ll_thresh:
#            plt.plot(v_lon,v_lat,color='g',linewidth=0.3)

    ## Loading coastline polygon.
    # For visualisation purpose, delete this part if you do not have the coastline
    # shapfile
    coastline_filename = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/"\
                         + "coastlines-split-4326/streetmap_coastline_Bretagne.pkl"

    try:
        with open(coastline_filename, 'rb') as f:
            l_coastline_poly = pickle.load(f)
    except:
        with open(coastline_filename, 'rb') as f:
            l_coastline_poly = pickle.load(f, encoding='latin1')
    for point in l_coastline_poly:
        poly = np.array(point)
        plt.plot(poly[:,0],poly[:,1],color="k",linewidth=0.8)

    print("Detecting abnormal tracks in the test set...")
    cmap_anomaly = plt.cm.get_cmap('autumn')
    N_dict = len(l_dict)
    d_i = 0
    n_anomalies = 0
    for D in tqdm(l_dict):
        try:
            c = cmap_anomaly(float(d_i)/(N_dict-1))
        except:
            c = 'r'
        d_i += 1
        m_tar = D["seq"]
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        if len(m_tar) < config.min_duration:
            continue
        v_lat = (m_tar[:,0]/float(config.onehot_lat_bins))*LAT_RANGE + config.lat_min
        v_lon = (m_tar[:,1]-float(config.onehot_lat_bins))/config.onehot_lon_bins*LON_RANGE + config.lon_min
        v_logprob = np.concatenate((v_logprob,[ll_t]))
        ll_stable = np.array([np.mean(log_weights_np[2*6,:])])
        if config.mode == "superposition_stable":
            ll_track = ll_stable
        else:
            ll_track = ll_t
        if ll_track < config.ll_thresh:
#            plt.plot(v_lon,v_lat,color='r',linewidth=0.8)
            plt.plot(v_lon,v_lat,color=c,linewidth=1.2)
            n_anomalies +=1

    plt.xlim([config.lon_min,config.lon_max])
    plt.ylim([config.lat_min,config.lat_max])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
#    plt.title("Abnormal tracks in the test set (red)")
    plt.tight_layout()
    fig_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/" \
            + config.mode + "-"\
            + config.bound + "-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name)\
            + "-latent_size-" + str(config.latent_size)\
            + "-ll_thresh" + str(config.ll_thresh) + "-"\
            + "missing_data-" + str(config.missing_data)\
            + "n_anomalies-"+str(n_anomalies)\
            + ".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()

#===============================================================================
#===============================================================================    
elif config.mode == "traj_reconstruction":
    """ TRAJECTORY RECONSTRUCTION
    We delete a segment of 2 hours in each tracks (in the test set), then
    reconstruct this part by the information embedded in the Embedding block.
    """
    save_dir = "results/"\
                + config.trainingset_path.split("/")[-2] + "/"\
                + "traj_reconstruction-"\
                + os.path.basename(config.trainingset_name) + "-"\
                + os.path.basename(config.testset_name) + "-"\
                + "-latent_size-" + str(config.latent_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Reconstructing AIS tracks...")
    for d_i in tqdm(list(range(dataset_size))):
        tar, mmsi, dense_sample, ll_t, ll_tracks\
                            = sess.run([targets, mmsis, track_sample, ll_per_t, ll_acc])
        if len(tar) < config.min_duration:
            continue

        sparse_tar = np.nonzero(np.squeeze(tar))[1].reshape(-1,4)
        for d_i_sample in range(config.num_samples):
            ## Plot received messages by blue dot, missing messages by red dot,
            # starting position by green dot.
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(sparse_tar[:,1],sparse_tar[:,0],'bo')
            plt.plot(sparse_tar[-18:-6,1],sparse_tar[-18:-6,0],'ro')
            plt.plot(sparse_tar[0,1],sparse_tar[0,0],'go')
            plt.ylim([0,config.onehot_lat_bins])
            plt.xlim([config.onehot_lat_bins,config.onehot_lat_bins+config.onehot_lon_bins])
            # Zoom-in
            plt.subplot(2,1,2)
            plt.plot(sparse_tar[:,1],sparse_tar[:,0],'bo')
            plt.plot(sparse_tar[-18:-6,1],sparse_tar[-18:-6,0],'ro')
            plt.plot(sparse_tar[0,1],sparse_tar[0,0],'go')
            ## Reconstructed positions
            logit_lat = np.argmax(dense_sample[:,d_i_sample,0:config.onehot_lat_bins], axis = 1)
            logit_lon = np.argmax(dense_sample[:,d_i_sample,config.onehot_lat_bins:config.onehot_lat_bins+config.onehot_lon_bins], axis = 1) + config.onehot_lat_bins
            plt.plot(logit_lon[1:],logit_lat[1:],'b')
            plt.plot(logit_lon[-17:-5],logit_lat[-17:-5],'r')
            plt.xlim([np.min(sparse_tar[:,1]) - 5, np.max(sparse_tar[:,1]) + 5])
            plt.ylim([np.min(sparse_tar[:,0]) - 5, np.max(sparse_tar[:,0]) + 5])

            fig_name = str(d_i)+"_"+str(d_i_sample)+"_"+str(mmsi)+"_"+str(ll_t)+".png"
            plt.savefig(os.path.join(save_dir,fig_name))
            plt.close()

#===============================================================================
#===============================================================================
elif config.mode == "traj_speed":
    """ SAVE SPEED PATTERN OF ABNORMAL TRACKS
    Save the speed pattern of abnormal tracks detected by the global
    thresholding detector.
    """
    save_dirname = "results/"\
                    + config.trainingset_path.split("/")[-2] + "/"\
                    + "traj_speed-"\
                    + os.path.basename(config.trainingset_name) + "-"\
                    + os.path.basename(config.testset_name) + "-"\
                    + str(config.latent_size) + "-"\
                    + str(-config.ll_thresh) + "/"
    if not os.path.exists(save_dirname):
        os.makedirs(save_dirname)
    v_logprob = np.empty((0,))
    m_abnormals = []

    try:
        with open(outputs_path,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outputs_path,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")
    d_i = -1

    print("Detecting abnormal tracks...")
    for D in tqdm(l_dict):
        d_i += 1
        mmsi = D["mmsi"]
        m_tar = D["seq"]
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        if len(m_tar) < config.min_duration:
            continue
        v_lat = (m_tar[:,0]/float(config.onehot_lat_bins))*LAT_RANGE + config.lat_min
        v_lon = (m_tar[:,1]-float(config.onehot_lat_bins))/config.onehot_lon_bins*LON_RANGE + config.lon_min

        if (ll_t < config.ll_thresh):
#            plt.figure(figsize=(960*2.5/FIG_DPI, 640*2.5/FIG_DPI), dpi=FIG_DPI)
            plt.figure(figsize=(960*2.5/FIG_DPI, 800*2.5/FIG_DPI), dpi=FIG_DPI)
            plt.subplot(2,1,1)
            plt.plot(v_lon,v_lat,'r')
#            v_logprob = np.concatenate((v_logprob,[ll_t]))
#            m_abnormals.append(m_tar[:,2]-(config.onehot_lat_bins+config.onehot_lon_bins))
            print("Log likelihood: ",ll_t)
            plt.xlim([config.lon_min,config.lon_max])
            plt.ylim([config.lat_min,config.lat_max])
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Abnormal track")
            plt.tight_layout()

            plt.subplot(2,1,2)
            v_x_axis = np.arange(len(m_tar))/6
            plt.plot(v_x_axis,m_tar[:,2]-(config.onehot_lat_bins+config.onehot_lon_bins),'ro')
            plt.ylim([0,30])
            plt.xlim([0,len(m_tar)/6])
            plt.ylabel("Speed over ground")
            plt.xlabel("Time (hour)")
            plt.tight_layout()
            fig_name = save_dirname + str(d_i)+ '_'+ str(int(mmsi))+'_'+str(ll_t)+'.png'
            plt.savefig(fig_name,dpi = FIG_DPI)
            plt.close()


