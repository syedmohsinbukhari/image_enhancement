#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:09:36 2017

@author: elcid
"""

import numpy as np
import cv2 as cv2
from os import listdir
from os import getcwd
from os.path import isdir


def main():
    img = cv2.imread('snowy_road.jpg', cv2.IMREAD_COLOR)
    print(img.shape)
    cv2.imshow('image', img)
    cv2.waitKey(5000) #wait 5 seconds for a keypress
    cv2.destroyAllWindows()
#    count_landscape = 0
#    count_portrait = 0
#    count_square = 0
#    
#    max_rows = (0,0, '')
#    max_cols = (0,0, '')
#    min_rows = (10000,10000, '')
#    min_cols = (10000,10000, '')
#    
#    for file in listdir(getcwd()):
#        if file.split('.')[-1] == 'py' or file.split('.')[-1] == 'txt':
#            continue
#        
#        img = cv2.imread(file, cv2.IMREAD_COLOR)
#        
#        
#        
#        if img.shape[0] > img.shape[1]:
#            count_portrait += 1
#        elif img.shape[0] < img.shape[1]:
#            count_landscape += 1
#        else:
#            count_square += 1
#        
#        if max_rows[0] < img.shape[0]:
#            max_rows = (img.shape[0], img.shape[1], file)
#        
#        if max_cols[1] < img.shape[1]:
#            max_cols = (img.shape[0], img.shape[1], file)
#        
#        if min_rows[0] > img.shape[0]:
#            min_rows = (img.shape[0], img.shape[1], file)
#        
#        if min_cols[1] > img.shape[1]:
#            min_cols = (img.shape[0], img.shape[1], file)
#        
#            
#    print('Number of landscape images: {0}\nNumber of portrait images: {1}\n'
#          'Number of square images: {2}\nTotal: {3}'.format(count_landscape, \
#                                    count_portrait, count_square, \
#                                    count_landscape+count_portrait+\
#                                    count_square))
#    
#    print('\nMax Rows: {0}'.format(max_rows))
#    print('Max Cols: {0}'.format(max_cols))
#    print('Min Rows: {0}'.format(min_rows))
#    print('Min Cols: {0}'.format(min_cols))


if __name__ == '__main__':
    main()