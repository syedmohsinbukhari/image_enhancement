#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:48:36 2017

@author: elcid
"""

import cv2 as cv2
from os import listdir
from os import getcwd
from os.path import isdir, join


def main():
    target_size = 128
    count = 0
    for file in listdir(join(getcwd(), 'original_img_data')):
        file_n = join(join(getcwd(),'original_img_data'), file)
        if file.split('.')[-1] == 'py' or file.split('.')[-1] == 'txt':
            continue
        
        if isdir(file_n):
            continue
        
        img = cv2.imread(file_n, cv2.IMREAD_COLOR)
        
        resize_factor = target_size / img.shape[0]
        if img.shape[0] > img.shape[1]:
            resize_factor = target_size / img.shape[1]
        
        img_resize = cv2.resize(img, None, fx=resize_factor, \
                                fy=resize_factor, \
                                interpolation = cv2.INTER_CUBIC)
        
        if file.split('.')[-1] == 'png':
            f_name = join(join(getcwd(), 'scaled_img_data'), str(count) + \
                          '.' + file.split('.')[-1])
        else:
            f_name = join(join(getcwd(), 'scaled_img_data'), str(count) + \
                          '.jpg')
        
        cv2.imwrite(f_name, img_resize)
        count += 1
        
        if count%100 == 0:
            print('count: ' + str(count))
            

if __name__ == '__main__':
    main()