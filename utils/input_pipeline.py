#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:15:16 2018

@author: elcid
"""

from os.path import join
from os import getcwd
from random import shuffle
import numpy as np
import cv2

def get_img_data(batch_size, file_name, CV_IMG_TYPE, IMG_CHN, IMG_DIM):

    f_name = join(join(getcwd(), 'img_data'), file_name)
    f = open(f_name, 'r+')

    f_rand = f.readlines()
    shuffle(f_rand)

    count = 0

    inp_dim = [batch_size, IMG_CHN]
    [inp_dim.append(i) for i in IMG_DIM]
    inp_dim = tuple(inp_dim)

    input_imgs = np.empty(inp_dim)
    output_imgs = np.empty(inp_dim)

    for line in f_rand:
        input_img, output_img = line.split(',')
        output_img = output_img.splitlines()[0]

        input_img_arr = cv2.imread(input_img, CV_IMG_TYPE)
        input_img_arr = input_img_arr/255.0

        if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
            input_img_arr = np.expand_dims(input_img_arr, axis=0)
        else:
            input_img_arr = np.swapaxes(input_img_arr, 0, 2)
            input_img_arr = np.swapaxes(input_img_arr, 1, 2)

        output_img_arr = cv2.imread(output_img, CV_IMG_TYPE)
        output_img_arr = output_img_arr/255.0

        if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
            output_img_arr = np.expand_dims(output_img_arr, axis=0)
        else:
            output_img_arr = np.swapaxes(output_img_arr, 0, 2)
            output_img_arr = np.swapaxes(output_img_arr, 1, 2)

        input_imgs[count%batch_size] = input_img_arr
        output_imgs[count%batch_size] = output_img_arr

        count += 1
        if count%batch_size == 0:
            yield input_imgs, output_imgs

    yield input_imgs, output_imgs