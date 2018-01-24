#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:13:52 2018

@author: elcid
"""

import tensorflow as tf

def conv2d(inputs, filters, kernel_size, name, strides=1, padding='same'):
    with tf.variable_scope(name):
        layer = tf.layers.conv2d(inputs=inputs,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 data_format='channels_first')
        return layer

def conv2d_t(inputs, filters, kernel_size, name, strides=2, padding='same'):
    with tf.variable_scope(name):
        layer = tf.layers.conv2d_transpose(inputs=inputs,
                                           filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format='channels_first')
        return layer

def max_pool2d(inputs, name, pool_size=2, strides=2, padding='same'):
    with tf.variable_scope(name):
        layer = tf.layers.max_pooling2d(inputs=inputs,
                                        pool_size=pool_size,
                                        strides=strides,
                                        padding=padding,
                                        data_format='channels_first')
        return layer

def dense(inputs, units, name):
    with tf.variable_scope(name):
        layer = tf.layers.dense(inputs=inputs,
                                units=units)
        return layer