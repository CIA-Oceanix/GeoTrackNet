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
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import logging
import scipy.special
import math

import runners
from flags_config import config, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
FIG_DPI = 150

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
LAT_RESO = config.anomaly_lat_reso
LON_RESO = config.anomaly_lon_reso
LAT_BIN = int(LAT_RANGE/LAT_RESO)
LON_BIN = int(LON_RANGE/LON_RESO)


## RUN TRAIN
#======================================

if config.mode == "train":
    print(config.dataset_path)
    fh = logging.FileHandler(os.path.join(config.logdir,config.log_filename+".log"))
    tf.logging.set_verbosity(tf.logging.INFO)
    # get TF logger
    logger = logging.getLogger('tensorflow')
    logger.addHandler(fh)
    runners.run_train(config)

## RUN TRAIN
#======================================
else:
    try:
        with open(config.testset_path,"rb") as f:
            Vs_test = pickle.load(f)
    except:
        with open(config.testset_path,"rb") as f:
            Vs_test = pickle.load(f, encoding='latin1')
    dataset_size = len(Vs_test)


if config.mode in ["save_outcomes","traj_reconstruction","log_density"]:
    tf.Graph().as_default()
    global_step = tf.train.get_or_create_global_step()
    inputs, targets, mmsis, lengths, model = runners.create_dataset_and_model(config,
                                                               config.split,
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

outcomes_save_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "outcomes-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name) + "-"\
            + str(config.latent_size)\
            + "-missing_data-" + str(config.missing_data)\
            + "-step-"+str(step)\
            + ".pkl"
if not os.path.exists(os.path.dirname(outcomes_save_name)):
    os.makedirs(os.path.dirname(outcomes_save_name))

if config.mode == "save_outcomes":
    """ SAVE_OUTCOMES
    Calculate and save log[p(x_t|h_t)] of each track in the test set.
    """
    l_dict = []
    for d_i in tqdm(list(range(math.ceil(dataset_size/config.batch_size)))):
        inp, tar, seq_len, mmsi, log_weights_np, true_np, ll_t =\
                 sess.run([inputs, targets, lengths, mmsis, log_weights, track_true, ll_per_t])
        for d_idx_inbatch in range(inp.shape[1]):
            D = dict()
            seq_len_d = seq_len[d_idx_inbatch]
            D["inp"] = np.nonzero(tar[:seq_len_d,d_idx_inbatch,:])[1].reshape(-1,4)
            D["mmsi"] = mmsi[d_idx_inbatch]
            D["log_weights"] = log_weights_np[:seq_len_d,:,d_idx_inbatch]
            l_dict.append(D)
    with open(outcomes_save_name,"wb") as f:
        pickle.dump(l_dict,f)

    """ LL
    Plot the distribution of log[p(x_t|h_t)] of each track in the test set.
    """

    v_ll = np.empty((0,))
    v_ll_stable = np.empty((0,))

    count = 0
    for D in tqdm(l_dict):
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        v_ll = np.concatenate((v_ll,[ll_t]))

    d_mean = np.mean(v_ll)
    d_std = np.std(v_ll)
    d_thresh = d_mean - 3*d_std

    plt.figure(figsize=(1920/FIG_DPI, 640/FIG_DPI), dpi=FIG_DPI)
    plt.plot(v_ll,'o')
    plt.title("Log likelihood " + os.path.basename(config.testset_name)\
              + ", mean = {0:02f}, std = {1:02f}, threshold = {2:02f}".format(d_mean, d_std, d_thresh))
    plt.plot([0,len(v_ll)], [d_thresh, d_thresh],'r')

    plt.xlim([0,len(v_ll)])
    fig_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/" \
            + "ll-" \
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

elif config.mode == "log_density":
    """ LOG DENSITY
    Calculate the mean and std map of log[p(x_t|h_t)].
    We divide the ROI into small cells, in each cell, we calculate the mean and
    the std of log[p(x_t|h_t)].
    """
    Map_ll = dict()
    for row  in range(LAT_BIN):
        for col in range(LON_BIN):
            Map_ll[ str(str(row)+","+str(col))] = []
    m_map_ll_std = np.zeros(shape=(LAT_BIN,LON_BIN))
    m_map_ll_mean = np.zeros(shape=(LAT_BIN,LON_BIN))
    m_map_density = np.zeros(shape=(LAT_BIN,LON_BIN))
    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))

    try:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")

    print("Calculatint ll map...")
    for D in tqdm(l_dict):
        tmp = D["inp"]
        log_weights_np = D["log_weights"]
        for d_timestep in range(2*6,len(tmp)):
            row = int(tmp[d_timestep,0]*0.01/LAT_RESO)
            col = int((tmp[d_timestep,1]-config.lat_bins)*0.01/LON_RESO)
            Map_ll[str(row)+","+str(col)].append(np.mean(log_weights_np[d_timestep,:]))

    def remove_gaussian_outlier(v_data,quantile=1.64):
        d_mean = np.mean(v_data)
        d_std = np.std(v_data)
        idx_normal = np.where(np.abs(v_data-d_mean)<=quantile*d_std)[0] #90%
        return v_data[idx_normal]

    for row  in range(LAT_BIN):
        for col in range(LON_BIN):
            v_cell = np.copy(Map_ll[str(row)+","+str(col)])
#            if len(v_cell) >1 and len(v_cell) < 5:
#                break
            v_cell = remove_gaussian_outlier(v_cell)
            m_map_ll_mean[row,col] = np.mean(v_cell)
            m_map_ll_std[row,col] = np.std(v_cell)
            m_map_density[row,col] = len(v_cell)

    save_dir = "results/"\
                + config.trainingset_path.split("/")[-2] + "/"\
                + "log_density-"\
                + os.path.basename(config.trainingset_name) + "-"\
                + os.path.basename(config.testset_name) + "-"\
                + str(config.latent_size) + "-"\
                + "missing_data-" + str(config.missing_data)\
                + "-step-"+str(step)\
                +"/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+"map_density-"+str(LAT_RESO)+"-"+str(LON_RESO),m_map_density)
    np.save(save_dir+"map_ll_mean-"+str(LAT_RESO)+"-"+str(LON_RESO),m_map_ll_mean)
    np.save(save_dir+"map_ll_std-"+str(LAT_RESO)+"-"+str(LON_RESO),m_map_ll_std)

    with open(os.path.join(save_dir,"map_ll-"+str(LAT_RESO)+"-"+str(LON_RESO)+".pkl"),"wb") as f:
        pickle.dump(Map_ll,f)


elif config.mode == "p_density":
    """ P DENSITY
    Calculate the mean and std map of p(x_t|h_t)
    We divide the ROI into small cells, in each cell, we calculate the mean and
    the std of p(x_t|h_t).
    """
    Map_ll = dict()
    for row  in range(LAT_BIN):
        for col in range(LON_BIN):
            Map_ll[ str(str(row)+","+str(col))] = []
    m_map_ll_std = np.zeros(shape=(LAT_BIN,LON_BIN))
    m_map_ll_mean = np.zeros(shape=(LAT_BIN,LON_BIN))
    m_map_density = np.zeros(shape=(LAT_BIN,LON_BIN))
    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))

    try:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")

    print("Calculatint ll map...")
    for D in tqdm(l_dict):
        tmp = D["inp"]
        log_weights_np = D["log_weights"]
        weights_np = np.exp(scipy.special.logsumexp(log_weights_np,axis=1))
        for d_timestep in range(2*6,len(tmp)):
            row = int(tmp[d_timestep,0]*0.01/LAT_RESO)
            col = int((tmp[d_timestep,1]-config.lat_bins)*0.01/LON_RESO)
            Map_ll[str(row)+","+str(col)].append(np.mean(weights_np[d_timestep,:]))

    def remove_gaussian_outlier(v_data,quantile=1.64):
        d_mean = np.mean(v_data)
        d_std = np.std(v_data)
        idx_normal = np.where(np.abs(v_data-d_mean)<=quantile*d_std)[0] #90%
        return v_data[idx_normal]

    for row  in range(LAT_BIN):
        for col in range(LON_BIN):
            v_cell = np.copy(Map_ll[str(row)+","+str(col)])
#            if len(v_cell) >1 and len(v_cell) < 5:
#                break
            v_cell = remove_gaussian_outlier(v_cell)
            m_map_ll_mean[row,col] = np.mean(v_cell)
            m_map_ll_std[row,col] = np.std(v_cell)
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
    np.save(save_dir+"map_density-"+str(LAT_RESO)+"-"+str(LON_RESO),m_map_density)
    np.save(save_dir+"map_ll_mean-"+str(LAT_RESO)+"-"+str(LON_RESO),m_map_ll_mean)
    np.save(save_dir+"map_ll_std-"+str(LAT_RESO)+"-"+str(LON_RESO),m_map_ll_std)

    with open(os.path.join(save_dir,"map_ll-"+str(LAT_RESO)+"-"+str(LON_RESO)+".pkl"),"wb") as f:
        pickle.dump(Map_ll,f)


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
        v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
        v_lon = tmp[:,1]*LON_RANGE + LON_MIN
        plt.plot(v_lon,v_lat,color=c,linewidth=0.8)
#        plt.plot(v_lon,v_lat,color='b',linewidth=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Load the outcomes of the embedding layer
    try:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")

    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))
#    print("Plotting tracks in the test set...")
#    for D in tqdm(l_dict):
#        m_tar = D["inp"]
#        log_weights_np = D["log_weights"]
#        ll_t = np.mean(log_weights_np)
#        if len(m_tar) < config.min_duration:
#            continue
#        v_lat = (m_tar[:,0]/float(config.lat_bins))*LAT_RANGE + LAT_MIN
#        v_lon = (m_tar[:,1]-float(config.lat_bins))/config.lon_bins*LON_RANGE + LON_MIN
#        v_ll = np.concatenate((v_ll,[ll_t]))
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
        m_tar = D["inp"]
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        if len(m_tar) < config.min_duration:
            continue
        v_lat = (m_tar[:,0]/float(config.lat_bins))*LAT_RANGE + LAT_MIN
        v_lon = (m_tar[:,1]-float(config.lat_bins))/config.lon_bins*LON_RANGE + LON_MIN
        v_ll = np.concatenate((v_ll,[ll_t]))
        ll_stable = np.array([np.mean(log_weights_np[2*6,:])])
        if config.mode == "superposition_stable":
            ll_track = ll_stable
        else:
            ll_track = ll_t
        if ll_track < config.ll_thresh:
#            plt.plot(v_lon,v_lat,color='r',linewidth=0.8)
            plt.plot(v_lon,v_lat,color=c,linewidth=1.2)
            n_anomalies +=1

    plt.xlim([LON_MIN,LON_MAX])
    plt.ylim([LAT_MIN,LAT_MAX])
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
            plt.ylim([0,config.lat_bins])
            plt.xlim([config.lat_bins,config.lat_bins+config.lon_bins])
            # Zoom-in
            plt.subplot(2,1,2)
            plt.plot(sparse_tar[:,1],sparse_tar[:,0],'bo')
            plt.plot(sparse_tar[-18:-6,1],sparse_tar[-18:-6,0],'ro')
            plt.plot(sparse_tar[0,1],sparse_tar[0,0],'go')
            ## Reconstructed positions
            logit_lat = np.argmax(dense_sample[:,d_i_sample,0:config.lat_bins], axis = 1)
            logit_lon = np.argmax(dense_sample[:,d_i_sample,config.lat_bins:config.lat_bins+config.lon_bins], axis = 1) + config.lat_bins
            plt.plot(logit_lon[1:],logit_lat[1:],'b')
            plt.plot(logit_lon[-17:-5],logit_lat[-17:-5],'r')
            plt.xlim([np.min(sparse_tar[:,1]) - 5, np.max(sparse_tar[:,1]) + 5])
            plt.ylim([np.min(sparse_tar[:,0]) - 5, np.max(sparse_tar[:,0]) + 5])

            fig_name = str(d_i)+"_"+str(d_i_sample)+"_"+str(mmsi)+"_"+str(ll_t)+".png"
            plt.savefig(os.path.join(save_dir,fig_name))
            plt.close()

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
    v_ll = np.empty((0,))
    m_abnormals = []

    try:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f)
    except:
        with open(outcomes_save_name,"rb") as f:
            l_dict = pickle.load(f, encoding = "latin1")
    d_i = -1

    print("Detecting abnormal tracks...")
    for D in tqdm(l_dict):
        d_i += 1
        mmsi = D["mmsi"]
        m_tar = D["inp"]
        log_weights_np = D["log_weights"]
        ll_t = np.mean(log_weights_np)
        if len(m_tar) < config.min_duration:
            continue
        v_lat = (m_tar[:,0]/float(config.lat_bins))*LAT_RANGE + LAT_MIN
        v_lon = (m_tar[:,1]-float(config.lat_bins))/config.lon_bins*LON_RANGE + LON_MIN

        if (ll_t < config.ll_thresh):
#            plt.figure(figsize=(960*2.5/FIG_DPI, 640*2.5/FIG_DPI), dpi=FIG_DPI)
            plt.figure(figsize=(960*2.5/FIG_DPI, 800*2.5/FIG_DPI), dpi=FIG_DPI)
            plt.subplot(2,1,1)
            plt.plot(v_lon,v_lat,'r')
#            v_ll = np.concatenate((v_ll,[ll_t]))
#            m_abnormals.append(m_tar[:,2]-(config.lat_bins+config.lon_bins))
            print("Log likelihood: ",ll_t)
            plt.xlim([LON_MIN,LON_MAX])
            plt.ylim([LAT_MIN,LAT_MAX])
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Abnormal track")
            plt.tight_layout()

            plt.subplot(2,1,2)
            v_x_axis = np.arange(len(m_tar))/6
            plt.plot(v_x_axis,m_tar[:,2]-(config.lat_bins+config.lon_bins),'ro')
            plt.ylim([0,30])
            plt.xlim([0,len(m_tar)/6])
            plt.ylabel("Speed over ground")
            plt.xlabel("Time (hour)")
            plt.tight_layout()
            fig_name = save_dirname + str(d_i)+ '_'+ str(int(mmsi))+'_'+str(ll_t)+'.png'
            plt.savefig(fig_name,dpi = FIG_DPI)
            plt.close()


