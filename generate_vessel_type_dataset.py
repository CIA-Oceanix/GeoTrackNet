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
A script to generate data for the Vessel type identification block. 
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

import runners


### Bretagne
#LAT_MIN = 47.0
#LAT_MAX = 50.0
#LON_MIN = -7.0
#LON_MAX = -4.0


## Gulf of Mexico
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87


LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
SPEED_MAX = 30.0  # knots
FIG_DPI = 300

# Shared flags.
tf.app.flags.DEFINE_string("mode", "visualisation",
                           "The mode of the binary. "
                           "'save_outcomes','ll','log_density','visualisation'"
                           "'traj_reconstruction'.")

tf.app.flags.DEFINE_string("bound", "elbo",
                           "The bound to optimize. Can be 'elbo', or 'fivo'.")

tf.app.flags.DEFINE_integer("latent_size", 400,
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
tf.app.flags.DEFINE_float("ll_thresh", -17.47,
                          "Log likelihood for the anomaly detection.")

# Resolution flags.
tf.app.flags.DEFINE_integer("lat_bins", 350,
                            "Number of bins of the lat one-hot vector")
tf.app.flags.DEFINE_integer("lon_bins", 1050,
                            "Number of bins of the lon one-hot vector")
tf.app.flags.DEFINE_integer("sog_bins", 30,
                            "Number of bins of the sog one-hot vector")
tf.app.flags.DEFINE_integer("cog_bins", 72,
                            "Number of bins of the cog one-hot vector")

tf.app.flags.DEFINE_float("anomaly_lat_reso", 0.1,
                          "Lat resolution for anomaly detection.")
tf.app.flags.DEFINE_float("anomaly_lon_reso", 0.1,
                          "Lon resolution for anomaly detection.")

# Dataset flags
tf.app.flags.DEFINE_string("dataset", "MarineC",
                           "Dataset. Can be 'Brittany' or 'MarineC'.")
tf.app.flags.DEFINE_string("trainingset_name", "MarineC_Jan2014_norm/MarineC_Jan2014_norm_train.pkl",
                           "Path to load the trainingset from.")
tf.app.flags.DEFINE_string("testset_name", "MarineC_Jan2014_norm/MarineC_Jan2014_norm_train.pkl",
                           "Path to load the testset from.")  
tf.app.flags.DEFINE_string("split", "train",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")  
tf.app.flags.DEFINE_boolean("missing_data", True,
                           "If true, a part of input track will be deleted.")  


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


FLAGS = tf.app.flags.FLAGS
config = FLAGS
config.data_dim  = config.lat_bins + config.lon_bins\
                 + config.sog_bins + config.cog_bins # error with data_dimension


### SC-PC-086    
#if config.dataset == "Brittany":
#    config.dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/mt314/"
#elif config.dataset == "MarineC":
#    config.dataset_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/"
#else:
#    raise ValueError("Unkown dataset (must be 'Brittany' or 'MarineC'.")

### Other PCs  
if config.dataset == "Brittany":
    config.dataset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/mt314/"
elif config.dataset == "MarineC":
    config.dataset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/"
else:
    raise ValueError("Unkown dataset (must be 'Brittany' or 'MarineC'.")
    
    
# TESTSET_PATH
if config.testset_name == "":
    config.testset_name = config.trainingset_name.replace("_train","_test")
config.trainingset_path = config.dataset_path + config.trainingset_name
config.testset_path = config.dataset_path + config.testset_name
# lazy reason
config.dataset_path = config.testset_path

print("Training set: " + config.trainingset_path)
print("Test set: " + config.testset_path)

with open(config.testset_path,"rb") as f:
    Vs_test = pickle.load(f)
dataset_size = len(Vs_test)

config.min_duration *= 6 # converting from hour to sequence length

# LOG DIR
logdir_name = "/" + config.bound + "-"\
             + os.path.basename(config.trainingset_name)\
             + "-data_dim-" + str(config.data_dim)\
             + "-latent_size-" + str(config.latent_size)\
             + "-batch_size-50"
config.logdir = config.log_dir + logdir_name
if not os.path.exists(config.logdir):
    raise ValueError(config.logdir + " doesnt exist")

LAT_RESO = config.anomaly_lat_reso
LON_RESO = config.anomaly_lon_reso
LAT_BIN = int(LAT_RANGE/LAT_RESO)
LON_BIN = int(LON_RANGE/LON_RESO)

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
#
if config.mode == "traj_reconstruction":
    missing_data = True
#else:
#    missing_data = False

track_sample, track_true, log_weights, ll_per_t, ll_acc, \
        rnn_state_tf, rnn_latent_tf, rnn_out_tf = runners.create_eval_graph(inputs, targets,
                                                           lengths, model, config)
                                    
                                    
                                    
saver = tf.train.Saver()
sess = tf.train.SingularMonitoredSession()
import matplotlib.pyplot as plt

runners.wait_for_checkpoint(saver, sess, config.logdir) 
step = sess.run(global_step)
#print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


###############################################################################
import csv
l_zone = [14,15,16]
dataset_vessel_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Zone{}_2014_01_Vessel.csv"
VesselTypes = dict()
v_vessel_types = []
for zone in l_zone:
    with open(dataset_vessel_path.format(zone), "rb") as f:
        reader = csv.reader(f)
        v_labels = reader.next()
        for row in reader:
            try:
                mmsi_ = int(row[0])
                type_ = int(row[4])
                v_vessel_types.append([mmsi_, type_])
                if  mmsi_ not in VesselTypes.keys():
                    VesselTypes[mmsi_] = [type_]
                elif type_ not in VesselTypes[mmsi_]:
                    VesselTypes[mmsi_].append(type_)
            except:
                continue
v_vessel_types = np.array(v_vessel_types).astype(np.int)
for mmsi_ in VesselTypes.keys():
    VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])

np.count_nonzero(np.logical_and(v_vessel_types[:,1] >= 80, v_vessel_types[:,1] <= 89))
np.count_nonzero(v_vessel_types[:,1] == 31)

def sublist(lst1, lst2):
   ls1 = [element for element in lst1 if element in lst2]
   ls2 = [element for element in lst2 if element in lst1]
   return (len(ls1) != 0) and (ls1 == ls2)
###############################################################################


l_vessel_type_dataset = []
for d_i in xrange(dataset_size):
    mmsi, rnn_state_np, rnn_latent_np, rnn_out_np, ll_acc_np = \
            sess.run([mmsis, rnn_state_tf, rnn_latent_tf, rnn_out_tf, ll_acc])
    print(d_i)
    d_i_max_ll = np.argmax(ll_acc_np) # indice of the sample with max likelihood
    mmsi_ = int(mmsi)
    if len(rnn_state_np) != 144:
        continue
    try:
        tmp = dict()
        if sublist(VesselTypes[mmsi_], range(70,80)): # cargo
            tmp['vessel_type'] = 0
        elif sublist(VesselTypes[mmsi_], range(60,70)): # passenger
            tmp['vessel_type'] = 1
        elif sublist(VesselTypes[mmsi_], range(80,90)): # tanker
            tmp['vessel_type'] = 2
        elif sublist(VesselTypes[mmsi_], [31,32,52]): # tug
            tmp['vessel_type'] = 3
        else:
            continue
        tmp['mmsi'] = mmsi
        # keep only the sample with max log likelihood
        tmp['rnn_state'] = np.squeeze(rnn_state_np[:,:,d_i_max_ll,:])
#        tmp['rnn_latent'] = np.squeeze(rnn_latent_np[:,:,d_i_max_ll,:])
#        tmp['rnn_out'] = np.squeeze(rnn_out_np[:,:,d_i_max_ll,:])
        
        # Because of the memory issue, we use only 50% of the tracks on the 
        # training set for this task. 
        token = np.random.rand()
        if token <= 0.5:
            l_vessel_type_dataset.append(tmp) 
        
    except:
        continue

import pickle
if config.dataset == "Brittany":
    save_path = "/users/local/dnguyen/Datasets/AIS_datasets/mt314/"
elif config.dataset == "MarineC":
    save_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/"
else:
    raise ValueError("Unkown dataset (must be 'Brittany' or 'MarineC'.")
    
rnn_save_name = os.path.join(save_path,config.testset_name.replace(".pkl","")+"_rnn_state.pkl")
with open(rnn_save_name,"wb") as f:
    pickle.dump(l_vessel_type_dataset, f) #2392
