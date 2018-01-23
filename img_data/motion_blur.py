#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:24:17 2017

@author: elcid
"""

import cv2 as cv2
from os import listdir
from os import getcwd
from os.path import isdir, join
import numpy as np

def main():
    count = 0
    size = 15
    
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    print(kernel_motion_blur)
    
    for file in listdir(join(getcwd(), 'scaled_cropped_img_data')):
        if file.split('.')[-1] == 'py' or file.split('.')[-1] == 'txt':
            continue
        
        if isdir(join(join(getcwd(), 'scaled_cropped_img_data'), file)):
            continue
        
        f_name_base = file.split('.')[0]
        f_name_ext = file.split('.')[-1]
        
        img = cv2.imread(join(join(getcwd(), 'scaled_cropped_img_data'), \
                              file), cv2.IMREAD_COLOR)
        
        # applying the kernel to the input image
        img_blured = cv2.filter2D(img, -1, kernel_motion_blur)
        
        f_name = join(join(getcwd(), 'blured_img_data'), f_name_base + '_.' + \
                      f_name_ext)
        cv2.imwrite(f_name, img_blured)
        
        count += 1
        
        if count%100 == 0:
            print('count: ' + str(count))
            

if __name__ == '__main__':
    main()