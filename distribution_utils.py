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
Distribution utils for MultitaskAIS. 
"""

#import numpy as np
import tensorflow as tf

def sample_one_hot(prob_):
    prob = prob_/tf.tile(tf.reduce_sum(prob_,axis=1,keep_dims=True),
                               multiples=[1,tf.shape(prob_)[1]])
    cdv = tf.cumsum(prob,axis = 1, exclusive=False)
    cdv_ex = tf.cumsum(prob,axis = 1, exclusive=True)
    
    thresh = tf.tile(tf.random_uniform(shape=(tf.shape(prob)[0],1)),
                        multiples=[1,tf.shape(prob)[1]])
    
    less_equal = tf.less_equal(cdv_ex,thresh)
    greater = tf.greater(cdv,thresh)
    
    one_hot = tf.cast(tf.logical_and(less_equal,greater), tf.float32)
    return one_hot

def sample_from_max_logits(logit,lat_bins, lon_bins, sog_bins, cog_bins):
    l_embedding_sizes = [lat_bins, lon_bins, sog_bins, cog_bins]
    logit_lat, logit_lon,logit_sog,logit_cog = tf.split(logit,l_embedding_sizes,axis=1)
    ind_lat = tf.argmax(logit_lat,axis = 1)
    ind_lon = tf.argmax(logit_lon,axis = 1)
    ind_sog = tf.argmax(logit_sog,axis = 1)
    ind_cog = tf.argmax(logit_cog,axis = 1)
    onehot_lat = tf.one_hot(ind_lat,lat_bins)
    onehot_lon = tf.one_hot(ind_lon,lon_bins)
    onehot_sog = tf.one_hot(ind_sog,sog_bins)
    onehot_cog = tf.one_hot(ind_cog,cog_bins)
    fourhot = tf.concat([onehot_lat,onehot_lon,onehot_sog,onehot_cog],axis = 1)
    return fourhot
    
def sample_from_logits(logit,lat_bins, lon_bins, sog_bins, cog_bins):
    l_embedding_sizes = [lat_bins, lon_bins, sog_bins, cog_bins]
    logit_lat, logit_lon,logit_sog,logit_cog = tf.split(logit,l_embedding_sizes,axis=1)
    dist_lat = tf.contrib.distributions.Bernoulli(logits=logit_lat)
    dist_lon = tf.contrib.distributions.Bernoulli(logits=logit_lon)
    dist_sog = tf.contrib.distributions.Bernoulli(logits=logit_sog)
    dist_cog = tf.contrib.distributions.Bernoulli(logits=logit_cog)
    sample_lat = dist_lat.sample()
    sample_lon = dist_lon.sample()
    sample_sog = dist_sog.sample()
    sample_cog = dist_cog.sample()
    sample_all = tf.concat([sample_lat,sample_lon,sample_sog,sample_cog],axis = 1)
    return sample_all
    
def sample_from_probs(probs_,lat_bins, lon_bins, sog_bins, cog_bins):
    l_embedding_sizes = [lat_bins, lon_bins, sog_bins, cog_bins]
    def squash_prob(l_old_prob):
        l_new_probs = []
        for old_prob in l_old_prob:
            new_probs0 = old_prob/tf.reshape(tf.reduce_max(old_prob,axis=1),(-1,1))
            new_probs1 = tf.where(tf.equal(new_probs0,1.), tf.ones(tf.shape(old_prob))*0.9999, new_probs0)
            l_new_probs.append(new_probs1)
        return l_new_probs

    prob_lat, prob_lon,prob_sog,prob_cog = squash_prob(tf.split(probs_,l_embedding_sizes,axis=1))
    sample_lat = sample_one_hot(prob_lat)
    sample_lon = sample_one_hot(prob_lon)
    sample_sog = sample_one_hot(prob_sog)
    sample_cog = sample_one_hot(prob_cog)
    sample_all = tf.concat([sample_lat,sample_lon,sample_sog,sample_cog],axis = 1)
    return sample_all
