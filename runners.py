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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import time

import numpy as np
import tensorflow as tf

import bounds
from data import datasets as datasets
from models import vrnn
import nested_utils as nested
import distribution_utils as dists

def create_eval_graph(inputs, targets, lengths, model, config):
    parallel_iterations=30
    swap_memory=True
    batch_size = tf.shape(lengths)[0]
    num_samples = config.num_samples
    max_seq_len = tf.reduce_max(lengths)
    init_states = model.zero_state(batch_size * num_samples, tf.float32)

    seq_mask = tf.transpose(
            tf.sequence_mask(lengths, maxlen=max_seq_len, dtype=tf.float32),
            perm=[1, 0])
    if num_samples > 1:
        inputs_tmp, seq_mask = nested.tile_tensors([(inputs,targets), seq_mask], [1, num_samples])
        inputs_ta, mask_ta = nested.tas_for_tensors([inputs_tmp, seq_mask], max_seq_len)
    else:
        inputs_ta, mask_ta = nested.tas_for_tensors([(inputs,targets), seq_mask], max_seq_len)

    t0 = tf.constant(0, tf.int32)
    init_states = model.zero_state(batch_size * num_samples, tf.float32)

    ta_names = ['log_weights_t','sampleds','trues','rnn_states','rnn_latents', 'rnn_outs']
    tas = [tf.TensorArray(tf.float32, max_seq_len, name='%s_ta' % n)
             for n in ta_names]

    log_weights_acc = tf.zeros([num_samples, batch_size], dtype=tf.float32)
    log_p_hat_acc = tf.zeros([batch_size], dtype=tf.float32)
    kl_acc = tf.zeros([num_samples * batch_size], dtype=tf.float32)
    if config.bound == "elbo":
        accs = (log_weights_acc, kl_acc)
    elif config.bound == "fivo":
        accs = (log_weights_acc, log_p_hat_acc, kl_acc)

    target_sampled0 = tf.zeros(shape = [batch_size*num_samples, config.data_dim],
                               dtype = tf.float32)
    target_true0 = tf.zeros(shape = [batch_size*num_samples, config.data_dim],
                               dtype = tf.float32)
    init_targets = (target_sampled0, target_true0)


    def while_predicate(t, *unused_args):
        return t < max_seq_len

    resampling_criterion=bounds.ess_criterion


    def while_step(t, rnn_state, tas, accs, while_samples):
        """Implements one timestep of IWAE computation."""
        if config.bound == "elbo":
            log_weights_acc, kl_acc = accs
        elif config.bound == "fivo":
            log_weights_acc, log_p_hat_acc, kl_acc = accs
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)

        if config.missing_data:
            cur_inputs = tf.cond(tf.logical_and(t < max_seq_len - 6,t >= max_seq_len - 18),
                                 lambda: while_samples,
                                 lambda: cur_inputs)

        # Run the cell for one step.
        log_q_z, log_p_z, log_p_x_given_z, kl, new_state, new_rnn_out, dists_return\
                                                        = model(cur_inputs,
                                                                rnn_state,
                                                                cur_mask,
                                                                return_value = "probs"
                                                                )

        new_sample0 = dists.sample_from_probs(dists_return,
                                              config.onehot_lat_bins,
                                              config.onehot_lon_bins,
                                              config.onehot_sog_bins,
                                              config.onehot_cog_bins)
        new_sample0 = tf.cast(new_sample0, tf.float32)
        new_sample_ = (new_sample0, tf.zeros_like(new_sample0, dtype = tf.float32))

        # Compute the incremental weight and use it to update the current
        # accumulated weight
        kl_acc += kl * cur_mask
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
        log_alpha = tf.reshape(log_alpha, [config.num_samples, batch_size])
        log_weights_acc += log_alpha
        # Calculate the effective sample size.
        ess_num = 2 * tf.reduce_logsumexp(log_weights_acc, axis=0)
        ess_denom = tf.reduce_logsumexp(2 * log_weights_acc, axis=0)
        log_ess = ess_num - ess_denom
        if config.bound == "fivo":
            # Calculate the ancestor indices via resampling. Because we maintain the
            # log unnormalized weights, we pass the weights in as logits, allowing
            # the distribution object to apply a softmax and normalize them.
            resampling_dist = tf.contrib.distributions.Categorical(
                                logits=tf.transpose(log_weights_acc, perm=[1, 0]))
            ancestor_inds = tf.stop_gradient(
                    resampling_dist.sample(sample_shape=num_samples, seed=config.random_seed))
            # Because the batch is flattened and laid out as discussed
            # above, we must modify ancestor_inds to index the proper samples.
            # The particles in the ith filter are distributed every batch_size rows
            # in the batch, and offset i rows from the top. So, to correct the indices
            # we multiply by the batch_size and add the proper offset. Crucially,
            # when ancestor_inds is flattened the layout of the batch is maintained.
            offset = tf.expand_dims(tf.range(batch_size), 0)
            ancestor_inds = tf.reshape(ancestor_inds * batch_size + offset, [-1])
            noresample_inds = tf.range(num_samples * batch_size)
            # Decide whether or not we should resample; don't resample if we are past
            # the end of a sequence.
            should_resample = resampling_criterion(num_samples, log_ess, t)
            should_resample = tf.logical_and(should_resample,
                                         cur_mask[:batch_size] > 0.)
            float_should_resample = tf.to_float(should_resample)
            ancestor_inds = tf.where(
                                tf.tile(should_resample, [num_samples]),
                                ancestor_inds,
                                noresample_inds)
            new_state = nested.gather_tensors(new_state, ancestor_inds)
            new_sample_ = nested.gather_tensors(new_sample_, ancestor_inds)

        # Update the  Tensorarrays and accumulators.
        ta_updates = [log_alpha, new_sample_[0], new_sample_[1],
                                  new_state[0], new_state[1], new_rnn_out]
    #    ta_updates = [log_weights_acc, log_ess]
        new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]

        if config.bound == "fivo":
            # For the particle filters that resampled, update log_p_hat and
            # reset weights to zero.
            log_p_hat_update = tf.reduce_logsumexp(
                    log_weights_acc, axis=0) - tf.log(tf.to_float(num_samples))
            log_p_hat_acc += log_p_hat_update * float_should_resample
            log_weights_acc *= (1. - tf.tile(float_should_resample[tf.newaxis, :],
                                         [num_samples, 1]))
            new_accs = (log_weights_acc, log_p_hat_acc, kl_acc)
        elif config.bound == "elbo":
            new_accs = (log_weights_acc, kl_acc)
        return t + 1, new_state, new_tas, new_accs, new_sample_

    _, _, tas, accs, new_sample = tf.while_loop(while_predicate,
                                    while_step,
                                    loop_vars=(t0, init_states, tas, accs, init_targets),
                                    parallel_iterations=parallel_iterations,
                                    swap_memory=swap_memory)


    #log_weights, log_ess = [x.stack() for x in tas]
    log_weights, track_sample, track_true, \
            rnn_state_tf, rnn_latent_tf, rnn_out_tf = [x.stack() for x in tas]

    #log_weights, log_ess, resampled = [x.stack() for x in tas]
    if config.bound == "fivo":
        final_log_weights, log_p_hat, kl = accs
        # Add in the final weight update to log_p_hat.
        log_p_hat += (tf.reduce_logsumexp(final_log_weights, axis=0) -
                                  tf.log(tf.to_float(num_samples)))
        kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)
    elif config.bound == "elbo":
        final_log_weights, kl = accs
        log_p_hat = (tf.reduce_logsumexp(final_log_weights, axis=0) -
                                     tf.log(tf.to_float(num_samples)))
        kl = tf.reduce_mean(tf.reshape(kl, [num_samples, batch_size]), axis=0)

    ll_per_seq = log_p_hat
    ll_per_t = ll_per_seq / tf.to_float(lengths)
    #        ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
    #        ll_per_seq = tf.reduce_mean(ll_per_seq)
    return track_sample, track_true, log_weights, ll_per_t, \
            final_log_weights/tf.to_float(lengths), rnn_state_tf, rnn_latent_tf, rnn_out_tf

def create_dataset_and_model(config, shuffle, repeat):

    inputs, targets, mmsis, time_starts, time_ends, lengths, mean = datasets.create_AIS_dataset(config.testset_path,
                                                          os.path.join(os.path.dirname(config.trainingset_path),"mean.pkl"),
                                                          config.batch_size,
                                                          config.data_dim,
                                                          config.onehot_lat_bins,
                                                          config.onehot_lon_bins,
                                                          config.onehot_sog_bins,
                                                          config.onehot_cog_bins,
                                                          shuffle=shuffle,
                                                          repeat=repeat)
    # Convert the mean of the training set to logit space so it can be used to
    # initialize the bias of the generative distribution.
    generative_bias_init = -tf.log(1. / tf.clip_by_value(mean, 0.0001, 0.9999) - 1)
    generative_distribution_class = vrnn.ConditionalBernoulliDistribution
    model = vrnn.create_vrnn(inputs.get_shape().as_list()[2],
                             config.latent_size,
                             generative_distribution_class,
                             generative_bias_init=generative_bias_init,
                             raw_sigma_bias=0.5)
    return inputs, targets, mmsis, time_starts, time_ends, lengths, model


def restore_checkpoint_if_exists(saver, sess, logdir):
    """Looks for a checkpoint and restores the session from it if found.
    Args:
      saver: A tf.train.Saver for restoring the session.
      sess: A TensorFlow session.
      logdir: The directory to look for checkpoints in.
    Returns:
      True if a checkpoint was found and restored, False otherwise.
    """
    checkpoint = tf.train.get_checkpoint_state(logdir)
    if checkpoint:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        full_checkpoint_path = os.path.join(logdir, checkpoint_name)
        saver.restore(sess, full_checkpoint_path)
        return True
    return False


def wait_for_checkpoint(saver, sess, logdir):
    while True:
        if restore_checkpoint_if_exists(saver, sess, logdir):
            break
        else:
            tf.logging.info("Checkpoint not found in %s, sleeping for 60 seconds."
                      % logdir)
            time.sleep(60)


def run_train(config):

    def create_logging_hook(step, bound_value):
        """Creates a logging hook that prints the bound value periodically."""
        bound_label = config.bound + " bound"
        if config.normalize_by_seq_len:
            bound_label += " per timestep"
        else:
            bound_label += " per sequence"
        def summary_formatter(log_dict):
            return "Step %d, %s: %f" % (
                        log_dict["step"], bound_label, log_dict["bound_value"])
        logging_hook = tf.train.LoggingTensorHook(
                                {"step": step,
                                 "bound_value": bound_value},
                                every_n_iter=config.summarize_every,
                                formatter=summary_formatter)
        return logging_hook

    def create_loss():
        """Creates the loss to be optimized.
        Returns:
            bound: A float Tensor containing the value of the bound that is
                   being optimized.
            loss: A float Tensor that when differentiated yields the gradients
                to apply to the model. Should be optimized via gradient descent.
        """
        inputs, targets, mmsis, time_starts, time_ends, lengths, model = create_dataset_and_model(config,
                                                               shuffle=True,
                                                               repeat=True)
        # Compute lower bounds on the log likelihood.
        if config.bound == "elbo":
            ll_per_seq, _, _, _ = bounds.elbo(model,
                                              (inputs, targets),
                                              lengths,
                                              num_samples=1)
        elif config.bound == "fivo":
            ll_per_seq, _, _, _, _ = bounds.fivo(model,
                                                 (inputs, targets),
                                                 lengths,
                                                 num_samples=config.num_samples,
                                                 resampling_criterion=bounds.ess_criterion)
        # Compute loss scaled by number of timesteps.
        ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
        ll_per_seq = tf.reduce_mean(ll_per_seq)

        tf.summary.scalar("train_ll_per_seq", ll_per_seq)
        tf.summary.scalar("train_ll_per_t", ll_per_t)

        if config.normalize_by_seq_len:
            return ll_per_t, -ll_per_t
        else:
            return ll_per_seq, -ll_per_seq

    def create_graph():
        """Creates the training graph."""
        global_step = tf.train.get_or_create_global_step()
        bound, loss = create_loss()
        opt = tf.train.AdamOptimizer(config.learning_rate)
        grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())
        train_op = opt.apply_gradients(grads, global_step=global_step)
        return bound, train_op, global_step

    device = tf.train.replica_device_setter(ps_tasks=config.ps_tasks)
    with tf.Graph().as_default():
        if config.random_seed: tf.set_random_seed(config.random_seed)
        with tf.device(device):
            bound, train_op, global_step = create_graph()
            log_hook = create_logging_hook(global_step, bound)
            start_training = not config.stagger_workers
            with tf.train.MonitoredTrainingSession(master=config.master,
                                                   is_chief=config.task == 0,
                                                   hooks=[log_hook],
                                                   checkpoint_dir=config.logdir,
                                                   save_checkpoint_secs=120,
                                                   save_summaries_steps=config.summarize_every,
                                                   log_step_count_steps=config.summarize_every) as sess:
                cur_step = -1
                while True:
                    if sess.should_stop() or cur_step > config.max_steps: break
                    if config.task > 0 and not start_training:
                        cur_step = sess.run(global_step)
                        tf.logging.info("task %d not active yet, sleeping at step %d" %
                            (config.task, cur_step))
                        time.sleep(30)
                        if cur_step >= config.task * 1000:
                            start_training = True
                    else:
                        _, cur_step = sess.run([train_op, global_step])
#                         _, cur_step = sess.run([train_op, global_step])
