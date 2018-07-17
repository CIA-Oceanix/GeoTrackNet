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

import runners as runners
import logging


## Shared flags.
tf.app.flags.DEFINE_string("mode", "train",
                           "The mode of the binary. Must be 'train' or 'eval'.")

tf.app.flags.DEFINE_string("bound", "elbo",
                           "The bound to optimize. Can be 'elbo', or 'fivo'.")

tf.app.flags.DEFINE_integer("latent_size", 100,
                            "The size of the latent state of the model.")

tf.app.flags.DEFINE_string("log_dir", "./chkpt",
                           "The directory to keep checkpoints and summaries in.")

tf.app.flags.DEFINE_integer("batch_size", 50,
                            "Batch size.")
tf.app.flags.DEFINE_integer("num_samples", 16,
                           "The number of samples (or particles) for multisample "
                           "algorithms.")

# Dataset flags.
tf.app.flags.DEFINE_string("dataset", "Brittany",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")
tf.app.flags.DEFINE_string("dataset_name", "dataset8/dataset8_train.pkl",
                           "Path to load the dataset from.")
tf.app.flags.DEFINE_string("split", "train",
                           "Split to evaluate the model on. Can be 'train', 'valid', or 'test'.")

# Resolution flags.
tf.app.flags.DEFINE_integer("lat_bins", 300,
                            "Length of the latitdue one-hot vectors.")
tf.app.flags.DEFINE_integer("lon_bins", 300,
                            "Length of the longitude one-hot vectors.")
tf.app.flags.DEFINE_integer("sog_bins", 30,
                            "Length of the SOG one-hot vectors.")
tf.app.flags.DEFINE_integer("cog_bins", 72,
                            "Length of the COG one-hot vectors.")

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
config = FLAGS
config.data_dim  = config.lat_bins + config.lon_bins + config.sog_bins + config.cog_bins

if config.dataset == "Brittany":
    config.dataset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/mt314/"\
                     + FLAGS.dataset_name
elif config.dataset == "MarineC":
    config.dataset_path = "/homes/vnguye04/Bureau/Sanssauvegarde/Datasets/MarineC/"\
                     + FLAGS.dataset_name
else:
    raise ValueError("Unkown dataset (must be 'Brittany' or 'MarineC'.")

logdir_name = "/" + config.bound + "-" + os.path.basename(config.dataset_name)\
             + "-data_dim-" + str(config.data_dim)\
             + "-latent_size-" + str(config.latent_size)\
             + "-batch_size-" + str(config.batch_size)
config.logdir = config.log_dir + logdir_name
if (config.split == "test"):
    config.dataset_path = config.dataset_path.replace("_train","_test")
if not os.path.exists(config.logdir):
    os.makedirs(config.logdir)

print(config.dataset_path)
fh = logging.FileHandler(config.logdir + logdir_name + '.log')

def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging._logger.addHandler(fh)
  if FLAGS.mode == "train":
    runners.run_train(config)
  elif FLAGS.mode == "eval":
    runners.run_eval(config)

if __name__ == "__main__":
  tf.app.run()
