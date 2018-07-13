# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
A script to run training for the Embedding layer of MultitaskAIS
The code is adapted from 
https://github.com/tensorflow/models/tree/master/research/fivo 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pickle

import runners as runners

LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87.0
LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
FIG_DPI = 300

# Shared flags.
tf.app.flags.DEFINE_string("mode", "save_outcomes",
                           "The mode of the binary. "
                           "'ll','superposition','traj_speed','traj_reconstruction',"
                           "'log_density', 'superposition_density', 'save_outcomes'.")

tf.app.flags.DEFINE_string("bound", "elbo",
                           "The bound to optimize. Can be 'elbo', or 'fivo'.")

tf.app.flags.DEFINE_integer("latent_size", 100,
                            "The size of the latent state of the model.")

tf.app.flags.DEFINE_string("log_dir", "./chkpt",
                           "The directory to keep checkpoints and summaries in.")

tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size.")
tf.app.flags.DEFINE_integer("min_duration", 4,
                            "Min duration (hour) of a vessel track")
tf.app.flags.DEFINE_integer("num_samples", 16,
                           "The number of samples (or particles) for multisample "
                           "algorithms.")
tf.app.flags.DEFINE_float("ll_thresh", -10.73,
                          "Log likelihood for the anomaly detection.")
tf.app.flags.DEFINE_float("anomaly_lat_reso", 0.1,
                          "Lat resolution for anomaly detection.")
tf.app.flags.DEFINE_float("anomaly_lon_reso", 0.1,
                          "Lon resolution for anomaly detection.")


tf.app.flags.DEFINE_string("split", "test",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")
tf.app.flags.DEFINE_string("trainingset_name", "dataset8/dataset8_train.pkl",
                           "Path to load the trainingset from.")
tf.app.flags.DEFINE_string("testset_name", "dataset8/dataset8_test_Uturn.pkl",
                           "Path to load the testset from.")    

tf.app.flags.DEFINE_string("model", "vrnn",
                           "Model choice. Currently only 'vrnn' is supported.")

tf.app.flags.DEFINE_integer("random_seed", None,
                            "A random seed for seeding the TensorFlow graph.")

# Training flags.

tf.app.flags.DEFINE_boolean("normalize_by_seq_len", True,
                            "If true, normalize the loss by the number of timesteps "
                            "per sequence.")
tf.app.flags.DEFINE_float("learning_rate", 0.0003,
                          "The learning rate for ADAM.")
tf.app.flags.DEFINE_integer("max_steps", int(1e9),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 100,
                            "The number of steps between summaries.")

# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Evaluation flags.


FLAGS = tf.app.flags.FLAGS
lat_bins = 300; lon_bins = 300; sog_bins = 30; cog_bins = 72
FLAGS.data_dim  = lat_bins + lon_bins + sog_bins + cog_bins # error with data_dimension
config = FLAGS

# LOG DIR
logdir_name = "/" + config.bound + "-"\
             + os.path.basename(config.trainingset_name)\
             + "-data_dim-" + str(config.data_dim)\
             + "-latent_size-" + str(config.latent_size)\
             + "-batch_size-50"
config.logdir = config.log_dir + logdir_name
if not os.path.exists(config.logdir):
    print(config.logdir + " doesnt exist")

# TESTSET_PATH
if config.testset_name == "":
    config.testset_name = config.trainingset_name.replace("_train","_test")
config.trainingset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/mt314/" + config.trainingset_name
config.testset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/mt314/" + config.testset_name
# lazy reason
config.dataset_path = config.testset_path

print("Training set: " + config.trainingset_path)
print("Test set: " + config.testset_path)

with open(config.testset_path,"rb") as f:
    Vs_test = pickle.load(f)
dataset_size = len(Vs_test)

config.min_duration *= 6 # converting from hour to sequence length

"""
run_eval()
#*************************************#
"""
tf.Graph().as_default()
global_step = tf.train.get_or_create_global_step()
inputs, targets, mmsis, lengths, model = runners.create_dataset_and_model(config, 
                                                           config.split,
                                                           shuffle=False,
                                                           repeat=False)

if config.mode == "traj_reconstruction":
    missing_data = True
else:
    missing_data = False

track_sample, track_true, log_weights, ll_per_t, ll_acc\
                                    = runners.create_eval_graph(inputs, targets,
                                                           lengths, model, config,
                                                           missing_data = missing_data)
saver = tf.train.Saver()
sess = tf.train.SingularMonitoredSession()
import matplotlib.pyplot as plt

runners.wait_for_checkpoint(saver, sess, config.logdir) 
step = sess.run(global_step)
#print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


outcomes_save_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "outcomes-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name) + "-"\
            + str(config.latent_size) + ".pkl"

if config.mode == "save_outcomes":
    l_dict = []
    for d_i in range(dataset_size):
        D = dict()
        print(d_i)
        inp, tar, mmsi, log_weights_np, sample_np, true_np, ll_t =\
                 sess.run([inputs, targets, mmsis, log_weights, track_sample, track_true, ll_per_t])
        D["inp"] = np.nonzero(tar[:,0,:])[1].reshape(-1,4)
        D["mmsi"] = mmsi
        D["log_weights"] = log_weights_np
        try: 
            D["samples"] = np.nonzero(sample_np[:,:,:])[2].reshape(-1,4)
        except:
            D["samples"] = np.nonzero(sample_np[:,:,:])
        l_dict.append(D)
    if not os.path.exists(os.path.dirname(outcomes_save_name)):
        os.makedirs(os.path.dirname(outcomes_save_name))
    with open(outcomes_save_name,"wb") as f:
        pickle.dump(l_dict,f)            

elif config.mode == "ll":
    ## LL
    ###########################################################################
    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))
    
    for d_i in range(int(dataset_size)):
        print(d_i)
        ll_t = sess.run(ll_per_t)
        v_ll = np.concatenate((v_ll,ll_t))
    
    d_mean = np.mean(v_ll)
    d_std = np.std(v_ll)
    d_thresh = d_mean - 2*d_std
    
    plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI)  
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
            + "-ll_thresh" + str(config.ll_thresh)\
            + ".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()

elif config.mode == "superposition":
    ## 	SUPERPOSITION OF ANOMALY DETECTION
    ###########################################################################
    
    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))

    with open(config.trainingset_path.replace("_test","_train"),"rb") as f:
        Vs = pickle.load(f)
#    with open(config.testset_path,"rb") as f:
#        Vs = pickle.load(f)

    plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI)  
    #cmap = plt.cm.get_cmap('YlGnBu')
    cmap = plt.cm.get_cmap('Blues')
    l_keys = Vs.keys()
    N = len(Vs)
    for d_i in range(N):
        key = l_keys[d_i]
        c = cmap(float(d_i)/(N-1))
        tmp = Vs[key]
        v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
        v_lon = tmp[:,1]*LON_RANGE + LON_MIN
        plt.plot(v_lon,v_lat,color=c,linewidth=0.3)
    
    for d_i in range(dataset_size):
        print(d_i)
        inp, tar, mmsi, midi_sample, midi_true, ll_t =\
                 sess.run([inputs, targets, mmsis, track_sample, track_true, ll_per_t])
        if len(inp) < config.min_duration:
            continue
        tmp = np.nonzero(np.squeeze(tar)[:,:1400])[1].reshape(-1,2)
        v_lat = (tmp[:,0]/350.)*LAT_RANGE + LAT_MIN
        v_lon = (tmp[:,1]-350)/1050.*LON_RANGE + LON_MIN
        v_ll = np.concatenate((v_ll,ll_t))
        if ll_t < config.ll_thresh:
            plt.plot(v_lon,v_lat,color='r',linewidth=1)
            v_ll = np.concatenate((v_ll,ll_t))
            v_mmsi = np.concatenate((v_mmsi,mmsi))
    
    plt.xlim([LON_MIN,LON_MAX])
    plt.ylim([LAT_MIN,LAT_MAX])        
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Abnormal tracks in the test set (red)")
    plt.tight_layout()  
    fig_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/" \
            + config.bound + "-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name)\
            + "-latent_size-" + str(config.latent_size)\
            + "-ll_thresh" + str(config.ll_thresh)\
            + ".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()

elif config.mode == "traj_speed":
    ## SAVE ABNORMAL TRAJECTORY AND SPEED
    ###########################################################################
    save_dirname = "results/"\
                    + config.trainingset_path.split("/")[-2] + "/"\
                    + "anomalies-"\
                    + os.path.basename(config.trainingset_name) + "-"\
                    + os.path.basename(config.testset_name) + "-"\
                    + str(config.latent_size) + "-"\
                    + str(-config.ll_thresh) + "/"
    if not os.path.exists(save_dirname):
        os.makedirs(save_dirname)
    v_ll = np.empty((0,))
    m_abnormals = []
    d_i = 0
    
    while (d_i < dataset_size):
        d_i += 1
        print(d_i)
        inp, tar, mmsi, midi_sample, midi_true, ll_t =\
                 sess.run([inputs, targets, mmsis, track_sample, track_true, ll_per_t])
        if len(inp) < config.min_duration:
            continue
        tmp = np.nonzero(np.squeeze(tar)[:,:1400])[1].reshape(-1,2)
        v_lat = (tmp[:,0]/350.)*LAT_RANGE + LAT_MIN
        v_lon = (tmp[:,1]-350)/1050.*LON_RANGE + LON_MIN
    #    v_ll = np.concatenate((v_ll,ll_t))
        if (ll_t < config.ll_thresh):
            plt.figure(figsize=(960*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI)   
            plt.subplot(2,1,1)
            plt.plot(v_lon,v_lat,'r')
            v_ll = np.concatenate((v_ll,ll_t))
            m_abnormals.append(np.nonzero(np.squeeze(tar))[-1].reshape(-1,4))
            print("Log likelihood: ",ll_t)
            plt.xlim([LON_MIN,LON_MAX])
            plt.ylim([LAT_MIN,LAT_MAX])        
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Abnormal tracks")
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.plot((m_abnormals[-1][:,2]-1400),'o')
            plt.ylim([0,30])
            plt.ylabel("Speed over ground")
            plt.tight_layout()
            fig_name = save_dirname + str(d_i)+ '_'+ str(int(mmsi))+'_'+str(ll_t)+'.png'
            plt.savefig(fig_name,dpi = FIG_DPI)
            plt.close()

elif config.mode == "traj_reconstruction":
    ## TRAJECTORY RECONSTRUCTION
    ###########################################################################    
    save_dir = "results/"\
                + config.trainingset_path.split("/")[-2] + "/"\
                + "traj_reconstruction-"\
                + os.path.basename(config.trainingset_name) + "-"\
                + os.path.basename(config.testset_name) + "-"\
                + str(config.latent_size) + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for d_i in range(dataset_size):
        print(d_i)
        tar, mmsi, dense_sample, ll_t, ll_tracks\
                            = sess.run([targets, mmsis, track_sample, ll_per_t, ll_acc])
        if len(inp) < config.min_duration:
            continue
        
        sparse_tar = np.nonzero(np.squeeze(tar))[1].reshape(-1,4)
    
        for d_i_sample in range(config.num_samples):
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(sparse_tar[:,1],sparse_tar[:,0],'b')
            plt.plot(sparse_tar[-12:-6,1],sparse_tar[-12:-6,0],'r')
            plt.plot(sparse_tar[0,1],sparse_tar[0,0],'go')
            plt.ylim([0,350])
            plt.xlim([350,1050])
            
            plt.subplot(2,1,2)
            plt.plot(sparse_tar[:,1],sparse_tar[:,0],'b')
            plt.plot(sparse_tar[-12:-6,1],sparse_tar[-12:-6,0],'r')
            plt.plot(sparse_tar[0,1],sparse_tar[0,0],'go')
            # Reconstructed positions
            logit_lat = np.argmax(dense_sample[:,d_i_sample,0:500], axis = 1)
            logit_lon = np.argmax(dense_sample[:,d_i_sample,500:1500], axis = 1) + 500
            plt.plot(logit_lon[1:],logit_lat[1:],'bo')
            plt.plot(logit_lon[-11:-5],logit_lat[-11:-5],'ro')
            plt.xlim([np.min(sparse_tar[:,1]) - 5, np.max(sparse_tar[:,1]) + 5])
            plt.ylim([np.min(sparse_tar[:,0]) - 5, np.max(sparse_tar[:,0]) + 5])
            
            fig_name = str(d_i)+"_"+str(d_i_sample)+"_"+str(mmsi)+"_"+str(ll_t)+".png"
            plt.savefig(os.path.join(save_dir,fig_name))
            plt.close()
    
    
elif config.mode == "log_density":
    ## 	SUPERPOSITION OF ANOMALY DETECTION
    ###########################################################################
    m_density_map = np.zeros(shape=(350,1050))
    m_ll_acc_map = np.zeros(shape=(350,1050))
    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))
    
    for d_i in range(dataset_size):
        print(d_i)
        inp, tar, mmsi, midi_sample, midi_true, ll_t =\
                 sess.run([inputs, targets, mmsis, track_sample, track_true, ll_per_t])
        if len(inp) < config.min_duration:
            continue
        tmp = np.nonzero(np.squeeze(tar)[:,:1400])[1].reshape(-1,2)
        row = int(np.mean(tmp[:,0]))
        col = int(np.mean(tmp[:,1]))-350
        m_density_map[row,col] += 1
        m_ll_acc_map[row,col] += ll_t
    m_log_mean_map = np.divide(m_ll_acc_map, 
                               m_density_map, 
                               out=np.zeros_like(m_ll_acc_map), 
                               where=m_density_map!=0)
    save_dir = "results/"\
                + config.trainingset_path.split("/")[-2] + "/"\
                + "log_density-"\
                + os.path.basename(config.trainingset_name) + "-"\
                + os.path.basename(config.testset_name) + "-"\
                + str(config.latent_size) + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir+"density_map",m_density_map)
    np.save(save_dir+"ll_acc_map",m_ll_acc_map)
    np.save(save_dir+"log_mean_map",m_log_mean_map)
    
elif config.mode == "superposition_density":
    ## 	SUPERPOSITION OF ANOMALY DETECTION
    ###########################################################################
    save_dir = "results/"\
            + config.trainingset_path.split("/")[-2] + "/"\
            + "log_density-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + str(config.latent_size) + "/"
    m_log_mean_map = np.load(save_dir+"log_mean_map.npy")

    with open(config.trainingset_path.replace("_test","_train"),"rb") as f:
        Vs = pickle.load(f)

    plt.figure(figsize=(1920*2/FIG_DPI, 640*2/FIG_DPI), dpi=FIG_DPI)  
    #cmap = plt.cm.get_cmap('YlGnBu')
    cmap = plt.cm.get_cmap('Blues')
    l_keys = Vs.keys()
    N = len(Vs)
    for d_i in range(N):
        key = l_keys[d_i]
        c = cmap(float(d_i)/(N-1))
        tmp = Vs[key]
        v_lat = tmp[:,0]*LAT_RANGE + LAT_MIN
        v_lon = tmp[:,1]*LON_RANGE + LON_MIN
        plt.plot(v_lon,v_lat,color=c,linewidth=0.3)
        
    v_ll = np.empty((0,))
    v_mmsi = np.empty((0,))
    
    for d_i in range(dataset_size):
        print(d_i)
        inp, tar, mmsi, midi_sample, midi_true, ll_t =\
                 sess.run([inputs, targets, mmsis, track_sample, track_true, ll_per_t])
        if len(inp) < config.min_duration:
            continue
        tmp = np.nonzero(np.squeeze(tar)[:,:1400])[1].reshape(-1,2)
        row = int(np.mean(tmp[:,0]))
        col = int(np.mean(tmp[:,1]))-350
        v_lat = (tmp[:,0]/350.)*LAT_RANGE + LAT_MIN
        v_lon = (tmp[:,1]-350)/1050.*LON_RANGE + LON_MIN
        v_ll = np.concatenate((v_ll,ll_t))
        if ll_t < (m_log_mean_map[row,col]+config.minus_log):
            plt.plot(v_lon,v_lat,color='r',linewidth=0.8)
            v_ll = np.concatenate((v_ll,ll_t))
            v_mmsi = np.concatenate((v_mmsi,mmsi))
        else:
            plt.plot(v_lon,v_lat,color='y',linewidth=0.8)
    plt.xlim([LON_MIN,LON_MAX])
    plt.ylim([LAT_MIN,LAT_MAX])        
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Abnormal tracks in the test set (red)")
    plt.tight_layout()  
    fig_name = "results/"\
            + config.trainingset_path.split("/")[-2] + "/" \
            + config.bound + "-"\
            + os.path.basename(config.trainingset_name) + "-"\
            + os.path.basename(config.testset_name)\
            + "-latent_size-" + str(config.latent_size)\
            + "-density_mode" + str(config.minus_log)\
            + ".png"
    plt.savefig(fig_name,dpi = FIG_DPI)
    plt.close()
