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
Flag configuration.
Adapted from the original script of FIVO.
"""

import os
import tensorflow as tf
import pickle

### Bretagne
LAT_MIN = 47.5
LAT_MAX = 49.5
LON_MIN = -7.0
LON_MAX = -4.0

## Gulf of Mexico
"""
LAT_MIN = 26.5
LAT_MAX = 30.0
LON_MIN = -97.5
LON_MAX = -87
"""

SPEED_MAX = 30.0  # knots
FIG_DPI = 300



# Shared flags.
tf.app.flags.DEFINE_string("mode", "save_outcomes",
                           "The mode of the binary. Must be 'train'"
                           "'save_outcomes','ll','log_density','visualisation'"
                           "'traj_reconstruction' or 'traj_speed'.")

tf.app.flags.DEFINE_string("bound", "elbo",
                           "The bound to optimize. Can be 'elbo', or 'fivo'.")

tf.app.flags.DEFINE_integer("latent_size", 100,
                            "The size of the latent state of the model.")

tf.app.flags.DEFINE_string("log_dir", "./chkpt",
                           "The directory to keep checkpoints and summaries in.")

tf.app.flags.DEFINE_integer("batch_size", 50,
                            "Batch size.")
tf.app.flags.DEFINE_integer("min_duration", 4,
                            "Min duration (hour) of a vessel track")
tf.app.flags.DEFINE_integer("num_samples", 16,
                           "The number of samples (or particles) for multisample "
                           "algorithms.")
tf.app.flags.DEFINE_float("ll_thresh", -17.47,
                          "Log likelihood for the anomaly detection.")

# Resolution flags.
tf.app.flags.DEFINE_integer("lat_bins", 200,
                            "Number of bins of the lat one-hot vector")
tf.app.flags.DEFINE_integer("lon_bins", 300,
                            "Number of bins of the lon one-hot vector")
tf.app.flags.DEFINE_integer("sog_bins", 30,
                            "Number of bins of the sog one-hot vector")
tf.app.flags.DEFINE_integer("cog_bins", 72,
                            "Number of bins of the cog one-hot vector")

tf.app.flags.DEFINE_float("anomaly_lat_reso", 0.1,
                          "Lat resolution for anomaly detection.")
tf.app.flags.DEFINE_float("anomaly_lon_reso", 0.1,
                          "Lon resolution for anomaly detection.")


tf.app.flags.DEFINE_float("interval_max", 2*3600,
                          "Maximum interval between two successive AIS messages (in second).")

# Dataset flags
tf.app.flags.DEFINE_string("dataset", "Brittany",
                           "Dataset. Can be 'Brittany' or 'MarineC'.")
tf.app.flags.DEFINE_string("trainingset_name", "ct_2017010203_10_20/ct_2017010203_10_20_train.pkl",
                           "Path to load the trainingset from.")
tf.app.flags.DEFINE_string("testset_name", "ct_2017010203_10_20/ct_2017010203_10_20_test.pkl",
                           "Path to load the testset from.")
tf.app.flags.DEFINE_string("split", "train",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")
tf.app.flags.DEFINE_boolean("missing_data", False,
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
tf.app.flags.DEFINE_integer("max_steps", int(80000),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer("summarize_every", 100,
                            "The number of steps between summaries.")

# A contrario detection flags.
tf.app.flags.DEFINE_float("contrario_eps", 1e-10,
                          "A contrario eps.")
tf.app.flags.DEFINE_boolean("print_log", False,
                            "If true, print the current state of the program to screen.")


# Distributed training flags.
tf.app.flags.DEFINE_string("master", "",
                           "The BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is used.")
tf.app.flags.DEFINE_boolean("stagger_workers", True,
                            "If true, bring one worker online every 1000 steps.")

# Fix tf >=1.8.0 flags bug
tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer("data_dim", 0, "Data dimension")
tf.app.flags.DEFINE_string('log_filename', '', 'Log filename')
tf.app.flags.DEFINE_string('logdir_name', '', 'Log dir name')
tf.app.flags.DEFINE_string('logdir', '', 'Log directory')
tf.app.flags.DEFINE_string('dataset_path', '', 'Dataset path')
tf.app.flags.DEFINE_string('trainingset_path', '', 'Training set path')
tf.app.flags.DEFINE_string('testset_path', '', 'Test set path')



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
    config.dataset_path = "./data/"
elif config.dataset == "MarineC":
    config.dataset_path = "./data/"
else:
    raise ValueError("Unkown dataset (must be 'Brittany' or 'MarineC'.")

# TESTSET_PATH
if config.mode == "train":
    config.testset_name = config.trainingset_name
elif config.testset_name == "":
    config.testset_name = config.trainingset_name.replace("_train","_test")
config.trainingset_path = config.dataset_path + config.trainingset_name
config.testset_path = config.dataset_path + config.testset_name
# lazy reason
config.dataset_path = config.testset_path

print("Training set: " + config.trainingset_path)
print("Test set: " + config.testset_path)

config.min_duration *= 6 # converting from hour to sequence length

# LOG DIR
config.logdir_name = "/" + config.bound + "-"\
             + os.path.basename(config.trainingset_name)\
             + "-data_dim-" + str(config.data_dim)\
             + "-latent_size-" + str(config.latent_size)\
             + "-batch_size-50"
config.logdir = config.log_dir + config.logdir_name
if not os.path.exists(config.logdir):
    if config.mode == "train":
        os.makedirs(config.logdir)
    else:
        raise ValueError(config.logdir + " doesnt exist")

if config.log_filename == "":
    config.log_filename = os.path.basename(config.logdir)
