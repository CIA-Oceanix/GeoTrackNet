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
from tqdm import tqdm

import runners
from flags_config import config, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
FIG_DPI = 300

LAT_RANGE = LAT_MAX - LAT_MIN
LON_RANGE = LON_MAX - LON_MIN
LAT_RESO = config.anomaly_lat_reso
LON_RESO = config.anomaly_lon_reso
LAT_BIN = int(LAT_RANGE/LAT_RESO)
LON_BIN = int(LON_RANGE/LON_RESO)

with open(config.testset_path,"rb") as f:
    Vs_test = pickle.load(f)
dataset_size = len(Vs_test)

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

runners.wait_for_checkpoint(saver, sess, config.logdir) 
step = sess.run(global_step)
#print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))


###############################################################################
import csv
l_zone = [14,15,16]
dataset_vessel_path = "/users/local/dnguyen/Datasets/AIS_datasets/MarineC/2014/01/Zone{}_2014_01_Vessel.csv"
VesselTypes = dict()
v_vessel_types = []
print("Loading vessel type list...")
for zone in tqdm(l_zone):
    with open(dataset_vessel_path.format(zone), "rb") as f:
        reader = csv.reader(f)
        v_labels = next(reader)
        for row in reader:
            try:
                mmsi_ = int(row[0])
                type_ = int(row[4])
                v_vessel_types.append([mmsi_, type_])
                if  mmsi_ not in list(VesselTypes.keys()):
                    VesselTypes[mmsi_] = [type_]
                elif type_ not in VesselTypes[mmsi_]:
                    VesselTypes[mmsi_].append(type_)
            except:
                continue
v_vessel_types = np.array(v_vessel_types).astype(np.int)
for mmsi_ in list(VesselTypes.keys()):
    VesselTypes[mmsi_] = np.sort(VesselTypes[mmsi_])

np.count_nonzero(np.logical_and(v_vessel_types[:,1] >= 80, v_vessel_types[:,1] <= 89))
np.count_nonzero(v_vessel_types[:,1] == 31)

def sublist(lst1, lst2):
   ls1 = [element for element in lst1 if element in lst2]
   ls2 = [element for element in lst2 if element in lst1]
   return (len(ls1) != 0) and (ls1 == ls2)
###############################################################################


l_vessel_type_dataset = []
print("Calculating hiddens regimes...")
for d_i in tqdm(list(range(dataset_size))):
    mmsi, rnn_state_np, rnn_latent_np, rnn_out_np, ll_acc_np = \
            sess.run([mmsis, rnn_state_tf, rnn_latent_tf, rnn_out_tf, ll_acc])
    d_i_max_ll = np.argmax(ll_acc_np) # indice of the sample with max likelihood
    mmsi_ = int(mmsi)
    if len(rnn_state_np) != 144:
        continue
    try:
        tmp = dict()
        if sublist(VesselTypes[mmsi_], list(range(70,80))): # cargo
            tmp['vessel_type'] = 0
        elif sublist(VesselTypes[mmsi_], range(60,70)): # passenger
            tmp['vessel_type'] = 1
        elif sublist(VesselTypes[mmsi_], list(range(80,90))): # tanker
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

print("Saving results...")
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
