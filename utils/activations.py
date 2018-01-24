#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:27:16 2018

@author: elcid
"""

import tensorflow as tf

def relu(x):
    return tf.nn.relu(x)

def lrele(x):
    return tf.nn.leaky_relu(x)

def tanh(x):
    return tf.nn.tanh(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)