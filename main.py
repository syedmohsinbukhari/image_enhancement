#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 19:22:53 2017

@author: elcid
"""

import tensorflow as tf
import numpy as np
import cv2 as cv2
from os.path import join, basename
from os import getcwd
#from random import shuffle

CV_IMG_TYPE = cv2.IMREAD_GRAYSCALE
IMG_DIM = [64, 64, 1]
IMG_SZ = IMG_DIM[0]*IMG_DIM[1]*IMG_DIM[2]
D_L1_SZ = 1000
G_L1_SZ = 8192
G_L2_SZ = 8192

class GAN:
    
    def __init__(self):
        
        #Discriminator Network
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SZ])
        
        self.D_W1 = tf.Variable(self.xavier_init([IMG_SZ, D_L1_SZ]))
        self.D_B1 = tf.Variable(tf.zeros(shape=[D_L1_SZ]))
        
        self.D_W2 = tf.Variable(self.xavier_init([D_L1_SZ, 1]))
        self.D_B2 = tf.Variable(tf.zeros(shape=[1]))
        
        self.D_vars = [self.D_W1, self.D_W2, self.D_B1, self.D_B2]
        
        #Generator Network
        self.Z = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SZ])
        
        self.G_W1 = tf.Variable(self.xavier_init([IMG_SZ, G_L1_SZ]))
        self.G_B1 = tf.Variable(tf.zeros(shape=[G_L1_SZ]))
        
        self.G_W2 = tf.Variable(self.xavier_init([G_L1_SZ, G_L2_SZ]))
        self.G_B2 = tf.Variable(tf.zeros(shape=[G_L2_SZ]))
        
        self.G_W3 = tf.Variable(self.xavier_init([G_L2_SZ, IMG_SZ]))
        self.G_B3 = tf.Variable(tf.zeros(shape=[IMG_SZ]))
        
        self.G_vars = [self.G_W1, self.G_W2, self.G_W3, self.G_B1, \
                       self.G_B2, self.G_B3]
        
        #Defining Loss Functions
        self.G_z = self.generator(self.Z)
        self.D_x, self.D_x_logit = self.discriminator(self.X)
        self.D_z, self.D_z_logit = self.discriminator(self.G_z)
        
#        self.D_loss=-tf.reduce_mean(tf.log(self.D_x)+tf.log(1.0 - self.D_z))
#        self.G_loss=-tf.reduce_mean(tf.log(self.D_z))
        
        # Alternative losses:
        # -------------------
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.D_x_logit, labels=tf.ones_like(self.D_x_logit)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.D_z_logit, labels=tf.zeros_like(self.D_z_logit)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.D_z_logit, labels=tf.ones_like(self.D_z_logit)))

        #Define optimizers
        self.D_optimizer = tf.train.AdamOptimizer().minimize(self.D_loss, \
                                                 var_list = self.D_vars)
        self.G_optimizer = tf.train.AdamOptimizer().minimize(self.G_loss, \
                                                 var_list = self.G_vars)
        
        
    def train(self, epochs, batch_size, tf_session):
        sess = tf_session
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            
            data_cur = self.get_data(batch_size)
            
            for d in data_cur:
                _, D_loss_cur = sess.run([self.D_optimizer, \
                                                    self.D_loss], \
                                          feed_dict={self.X: d[1], \
                                                     self.Z: d[0]})
                _, G_loss_cur = sess.run([self.G_optimizer, self.G_loss], \
                                          feed_dict={self.Z: d[0]})
                
            if i%1==0:
                print('Iter: {}'.format(i))
                print('D loss: {:.4}'.format(D_loss_cur))
                print('G_loss: {:.4}'.format(G_loss_cur))
                print()
        
    
    def test(self, tf_session):
        sess = tf_session
        
        f_name = join(join(getcwd(), 'img_data'), 'testing_file.txt')
        f = open(f_name, 'r+')
        
        output_path = join(join(getcwd(), 'img_data'), 'output_img_data')
        
        if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
            output_img = np.zeros((IMG_DIM[0],IMG_DIM[1]*3), \
                                  dtype=np.float32)
        elif CV_IMG_TYPE == cv2.IMREAD_COLOR:
            output_img = np.zeros((IMG_DIM[0],IMG_DIM[1]*3,IMG_DIM[2]), \
                                  dtype=np.float32)
        else:
            print('Incorrect image type')
            return
        
        for line in f:
            input_img_path, label_img_path = line.split(',')
            label_img_path = label_img_path.splitlines()[0]
            output_img_path = join(output_path, basename(label_img_path))
            
            input_img = cv2.imread(input_img_path, CV_IMG_TYPE)
            input_img = np.reshape(input_img, (1,IMG_SZ))
            input_img = input_img/255.0
            
            pred_img = sess.run(self.G_z, \
                                feed_dict={self.Z: input_img})            
            pred_img = pred_img - pred_img.min()
            pred_img = pred_img/max(1.0, pred_img.max())
            
            label_img = cv2.imread(label_img_path, CV_IMG_TYPE)
            label_img = label_img/255.0
            
            if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
                pred_img = np.reshape(pred_img, tuple(IMG_DIM[0:2]))
                input_img = np.reshape(input_img, tuple(IMG_DIM[0:2]))
                
                output_img[:,0:IMG_DIM[1]] = input_img
                output_img[:,(IMG_DIM[1]):(2*IMG_DIM[1])] = pred_img
                output_img[:,(2*IMG_DIM[1]):(3*IMG_DIM[1])] = label_img
                
                output_img = output_img * 255.0
                
                cv2.imwrite(output_img_path, output_img)
                
            elif CV_IMG_TYPE == cv2.IMREAD_COLOR:
                pred_img = np.reshape(pred_img, tuple(IMG_DIM))
                input_img = np.reshape(input_img, tuple(IMG_DIM))
                
                output_img[:,0:IMG_DIM[1],:] = input_img
                output_img[:,(IMG_DIM[1]):(2*IMG_DIM[1]),:] = pred_img
                output_img[:,(2*IMG_DIM[1]):(3*IMG_DIM[1]),:] = label_img
                
                output_img = output_img * 255.0
                
                cv2.imwrite(output_img_path, output_img)
            
            else:
                return


    def discriminator(self, x):
        D_L1_out = tf.nn.relu(tf.matmul(x, self.D_W1) + self.D_B1)
        D_L2_logit = tf.matmul(D_L1_out, self.D_W2) + self.D_B2
        D_L2_out = tf.nn.sigmoid(D_L2_logit)
        return D_L2_out, D_L2_logit
        
        
    def generator(self, z):
        G_L1_out = tf.nn.relu(tf.matmul(z, self.G_W1) + self.G_B1)
        G_L2_logit = tf.matmul(G_L1_out, self.G_W2) + self.G_B2
        G_L2_out = tf.nn.relu(G_L2_logit)
        G_L3_logit = tf.matmul(G_L2_out, self.G_W3) + self.G_B3
        G_L3_out = tf.nn.sigmoid(G_L3_logit)
        return G_L3_out
        
        
    def xavier_init(self, size):
        std_dev = 1.0/tf.sqrt(size[0]/2.0)
        return tf.random_normal(shape=size, stddev=std_dev)
        
    
    def get_data(self, batch_size):
        
        f_name = join(join(getcwd(), 'img_data'), 'training_file.txt')
        f = open(f_name, 'r+')
        
#        f_rand = f.readlines()
#        shuffle(f_rand)
        
        count = 0
        input_imgs = np.empty((batch_size, IMG_SZ))
        output_imgs = np.empty((batch_size, IMG_SZ))
        for line in f:
            input_img, output_img = line.split(',')
            output_img = output_img.splitlines()[0]
            
            input_img_arr = cv2.imread(input_img, CV_IMG_TYPE)
            input_img_arr = np.reshape(input_img_arr, IMG_SZ)
            input_img_arr = input_img_arr/255.0
            output_img_arr = cv2.imread(output_img, CV_IMG_TYPE)
            output_img_arr = np.reshape(output_img_arr, IMG_SZ)
            output_img_arr = output_img_arr/255.0
            
            input_imgs[(count%batch_size),:] = input_img_arr
            output_imgs[(count%batch_size),:] = output_img_arr
            
            count += 1
            if count%batch_size == 0:
                yield input_imgs, output_imgs
        
        yield input_imgs, output_imgs
    
    
    
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    test_gan = GAN()
    test_gan.train(10, 10, sess)
    test_gan.test(sess)
    
    sess.close()


if __name__ == '__main__':
    main()