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
A set of utils for the a contrario anomaly detection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import operator as op
from functools import reduce

MAX_SEQUENCE_LENGTH = 4*6 #4 hours x 6 (time steps = 10 mins)
N_EVENT = 0 # number of event
for ns in range(1,MAX_SEQUENCE_LENGTH+1):
    n_ci = MAX_SEQUENCE_LENGTH-ns+1
    N_EVENT += n_ci

def nCr(n, r):
    """Function calculates the number of combinations (n choose r)"""
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer//denom

def nonzero_segments(x_):
    """Return list of consecutive nonzeros from x_"""
    run = []
    result = []
    for d_i in range(len(x_)):
        if x_[d_i] != 0:
            run.append(d_i)
        else:
            if len(run) != 0:
                result.append(run)
                run = []
    if len(run) != 0:
        result.append(run)
        run = []
    return result

def zero_segments(x_):
    """Return list of consecutive zeros from x_"""
    run = []
    result = []
    for d_i in range(len(x_)):
        if x_[d_i] == 0:
            run.append(d_i)
        else:
            if len(run) != 0:
                result.append(run)
                run = []
    if len(run) != 0:
        result.append(run)
        run = []
    return result

def NFA(ns,k):
    """Number of False Alarms"""
    B = 0
    for t in range(k,ns+1):
        B += nCr(ns,t)*(0.1**t)*(0.9**(ns-t))
    return 300*B


def contrario_detection(v_A_,epsilon=0.0091):
    """
    A contrario detection algorithms
    INPUT:
        v_A_: abnormal point indicator vector
        epsilon: threshold
    OUTPUT:
        v_anomalies: abnormal segment indicator vector

    """
    v_anomalies = np.zeros(len(v_A_))
    max_seq_len = min(MAX_SEQUENCE_LENGTH, len(v_A_))
    for d_ns in range(max_seq_len,0,-1):
        for d_ci in range(max_seq_len+1-d_ns):
            v_xi = v_A_[d_ci:d_ci+d_ns]
            d_k_xi = int(np.count_nonzero(v_xi))
            if NFA(d_ns,d_k_xi)<epsilon:
                v_anomalies[d_ci:d_ci+d_ns] = 1
    return v_anomalies

