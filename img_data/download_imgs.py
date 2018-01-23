#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:19:51 2017

@author: elcid
"""

# This file downloads images from URLs placed in a txt file in the same folder

from urllib.request import urlopen
from shutil import copyfileobj
from os.path import isfile
from os import stat
from os import remove
from os import listdir
from os import getcwd
from time import sleep

MIN_FILE_SIZE = 8192
MAX_FILE_SIZE = 8388608

def download_file(url, file_name):
    with urlopen(url) as response, open(file_name, 'wb') as out_file:
        if response.code == 200:
            try:
                copyfileobj(response, out_file)
                print('saving file: {0}, {1}KB'.format(file_name, \
                      round(stat(file_name).st_size/1024)))
            except:
                print('can not save this file: {0}, {1}KB'.format(\
                      file_name, round(stat(file_name).st_size/1024)))


def remove_small_files():
    f_count = 0
    for file in listdir(getcwd()):
        file_ext = file.split('.')[-1]
        if file_ext == 'py' or file_ext == 'txt':
            print('not considering file: {0}'.format(file))
        else:
            if stat(file).st_size < MIN_FILE_SIZE:
                f_count += 1
                print('deleting file: {0}, {1}KB'.format(file, \
                      round(stat(file).st_size/1024)))
                remove(file)
    print('Total files removed: {0}'.format(f_count))
    
    
def remove_big_files():
    f_count = 0
    for file in listdir(getcwd()):
        file_ext = file.split('.')[-1]
        if file_ext == 'py' or file_ext == 'txt':
            print('not considering file: {0}'.format(file))
        else:
            if stat(file).st_size > MAX_FILE_SIZE:
                f_count += 1
                print('deleting file: {0}, {1}KB'.format(file, \
                      round(stat(file).st_size/1024)))
                remove(file)
    print('Total files removed: {0}'.format(f_count))
    

def main():
    rm_sm_files = 0 #set this to 1 to remove very small files
    rm_bg_files = 0 #set this to 1 to remove very big files
    if rm_sm_files == 1:
        print('Removing small files')
        remove_small_files()
        return
    if rm_bg_files == 1:
        print('Removing big files')
        remove_big_files()
        return
        
    with open('img_list.txt', 'r') as img_list_file:
        f_count = 0
        for line in img_list_file:
            f_count += 1
            print(str(f_count) + ': ', end='')
            link = line.replace('\n', '')
            img_name = link.split('/')[-1]
            if not isfile(img_name):
                try:
                    download_file(link, img_name)
                except:
                    print('can not download this file: {0}'.format(img_name))
            else:
                print('file already exists: {0}, {1}KB'.format(img_name, \
                      round(stat(img_name).st_size/1024)))
            sleep(0.01)
    

if __name__ == '__main__':
    main()