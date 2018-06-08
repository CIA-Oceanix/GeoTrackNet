#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:42:00 2018

@author: vnguye04

A set of utils for the a contrario anomaly detection.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import operator as op

MAX_SEQUENCE_LENGTH = 4*6 #4 hours x 6 (time steps = 10 mins)
N_EVENT = 0 # number of event
for ns in range(1,MAX_SEQUENCE_LENGTH+1):
    n_ci = MAX_SEQUENCE_LENGTH-ns+1
    N_EVENT += n_ci


def nCr(n, r):
    """Function calculates the number of combinations (n choose r)"""
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
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
    for d_ns in range(L,0,-1):
        for d_ci in range(L+1-d_ns):
            v_xi = v_A_[d_ci:d_ci+d_ns]
            d_k_xi = int(np.count_nonzero(v_xi))
            if NFA(d_ns,d_k_xi)<epsilon:
                v_anomalies[d_ci:d_ci+d_ns] = 1
    return v_anomalies
    
