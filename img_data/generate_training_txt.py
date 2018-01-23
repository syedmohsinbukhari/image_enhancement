#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:38:41 2017

@author: elcid
"""

from os import listdir
from os import getcwd
from os.path import isdir, join


def main():
    count = 0
    f = open('training_file.txt', 'r+')
    
    for file in listdir(join(getcwd(),'blured_img_data')):
        if file.split('.')[-1] == 'py' or file.split('.')[-1] == 'txt':
            continue
        
        if isdir(join(join(getcwd(), 'blured_img_data'), file)):
            continue
        
        f_name_base = file.split('_')[0]
        f_name_ext = file.split('.')[-1]
        
        inp_file = join(join(getcwd(),'blured_img_data'), file)
        label_file = join(join(getcwd(),'scaled_cropped_img_data'), \
                          f_name_base+'.'+f_name_ext)
        f.write(inp_file+','+label_file+'\n')
        
        count += 1
        
        if count%100 == 0:
            print('count: ' + str(count))
    
    f.close()
            

if __name__ == '__main__':
    main()