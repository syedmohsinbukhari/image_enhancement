#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 21:55:28 2017

@author: elcid
"""

import cv2 as cv2
from os import listdir
from os import getcwd
from os.path import isdir, join


def main():
    target_size = 128
    count = 0
    for file in listdir(join(getcwd(), 'scaled_img_data')):
        if file.split('.')[-1] == 'py' or file.split('.')[-1] == 'txt':
            continue
        
        if isdir(join(join(getcwd(), 'scaled_img_data'), file)):
            continue
        
        img = cv2.imread(join(join(getcwd(), 'scaled_img_data'), file), \
                         cv2.IMREAD_COLOR)
        
        n_rows = img.shape[0]
        n_cols = img.shape[1]
        if n_rows%2 == 0:
            r_cen = n_rows/2
        else:
            r_cen = (n_rows-1)/2
        
        if n_cols%2 == 0:
            c_cen = n_cols/2
        else:
            c_cen = (n_cols-1)/2
            
        img_cropped = img[int(r_cen-(target_size/2)):\
                          int(r_cen+(target_size/2)), \
                          int(c_cen-(target_size/2)):\
                          int(c_cen+(target_size/2))]
        
        f_name = join(join(getcwd(), 'scaled_cropped_img_data'), file)
        
        cv2.imwrite(f_name, img_cropped)
        count += 1
        
        if count%100 == 0:
            print('count: ' + str(count))
            

if __name__ == '__main__':
    main()