# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:36:03 2017

@author: Syed Mohsin Bukhari
"""

import tensorflow as tf
import numpy as np
import cv2 as cv2
from os.path import join, isfile
from os import getcwd, unlink, listdir
from random import shuffle

CV_IMG_TYPE = cv2.IMREAD_GRAYSCALE
IMG_DIM = [64, 64, 1]
IMG_SZ = IMG_DIM[0]*IMG_DIM[1]*IMG_DIM[2]

class GAN:
    
    def __init__(self):
        
        #Defining Placeholders
        inp_dim = [None]
        [inp_dim.append(i) for i in IMG_DIM]
        self.X = tf.placeholder(dtype=tf.float32, shape=inp_dim)
        self.Z = tf.placeholder(dtype=tf.float32, shape=inp_dim)
        self.batch_size = 100
        
        #Get Generator and Discriminator Outputs
        self.G_z, self.G_z_mean, self.G_z_stddev = self.generator(self.Z)
        self.D_x, self.D_x_logit = self.discriminator(self.X, 'real')
        self.D_z, self.D_z_logit = self.discriminator(self.G_z, 'fake')
        
        #Defining Loss Functions
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.G_z_mean) + \
                           tf.square(self.G_z_stddev) - \
                           tf.log(tf.square(self.G_z_stddev) + 1e-8) - 1, 1)
        
        generated_flat = tf.reshape(self.G_z, \
                                    [self.batch_size, IMG_DIM[0]*IMG_DIM[1]])
        original_flat = tf.reshape(self.X, \
                                    [self.batch_size, IMG_DIM[0]*IMG_DIM[1]])
        self.generation_loss = -tf.reduce_sum(original_flat * tf.log(1e-8 + \
            generated_flat) + (1-original_flat) * tf.log(1e-8 + 1 - \
            generated_flat), 1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        
#        self.D_loss=-tf.reduce_mean(tf.log(self.D_x)+tf.log(1.0 - self.D_z))
#        self.G_loss=-tf.reduce_mean(tf.log(self.D_z))+self.latent_loss
        
        # Alternative losses:
        # -------------------
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.D_x_logit, labels=tf.ones_like(self.D_x_logit)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.D_z_logit, labels=tf.zeros_like(self.D_z_logit)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=self.D_z_logit, labels=tf.ones_like(self.D_z_logit)))
        
        #Prepare variable lists
        t_vars = tf.trainable_variables()
        self.D_vars = [var for var in t_vars if 'D_' in var.name]
        self.G_vars = [var for var in t_vars if 'G_' in var.name]
        
        #Define optimizers
        self.D_optimizer = tf.train.AdamOptimizer(learning_rate=0.001).\
                                                    minimize(self.D_loss, \
                                                 var_list = self.D_vars)
        self.G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).\
                                                    minimize(self.G_loss, \
                                                 var_list = self.G_vars)
        self.VAE_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).\
                                minimize(self.cost, var_list = self.G_vars)
        
        
    def train(self, epochs, tf_session):
        sess = tf_session
        sess.run(tf.global_variables_initializer())
        
        batch_size = self.batch_size
        
        for i in range(epochs):
            
            data_cur = self.get_data(batch_size, 'training_file.txt')
            
            for d in data_cur:
                for k in range(1):
                    _, D_loss_cur = sess.run([self.D_optimizer, 
                                              self.D_loss], \
                                              feed_dict={self.X: d[1], \
                                                         self.Z: d[0]})
                for k in range(1):
                    _, G_loss_cur = sess.run([self.G_optimizer, \
                                              self.G_loss], \
                                              feed_dict={self.Z: d[0]})
                for k in range(5):
                    _, VAE_loss_cur, generation_loss, latent_loss = \
                                            sess.run([self.VAE_optimizer, \
                                              self.cost,\
                                              self.generation_loss,\
                                              self.latent_loss], \
                                              feed_dict={self.X: d[1], \
                                                         self.Z: d[0]})
                
            if i%1==0:
                print('Iter: {}'.format(i))
                print('D loss: {0}'.format(D_loss_cur))
                print('G_loss: {0}'.format(G_loss_cur))
                print('mean_generation_loss: {0}'.format(np.mean(\
                      generation_loss)))
                print('mean_latent_loss: {0}'.format(np.mean(\
                      latent_loss)))
                print('VAE_loss: {0}'.format(VAE_loss_cur))
                print()
#                if (G_loss_cur<10.0):
#                    break
        
                
    def test(self, tf_session):
        sess = tf_session
        data_cur = self.get_data(self.batch_size, 'testing_file.txt')
        
        output_path = join(join(getcwd(), 'img_data'), 'output_img_data')
        
        #delete all previous files
        for the_file in listdir(output_path):
            file_path = join(output_path, the_file)
            try:
                if isfile(file_path):
                    unlink(file_path)
            except Exception as e:
                print(e)
        
        if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
            output_img = np.zeros((IMG_DIM[0],IMG_DIM[1]*3), \
                                  dtype=np.float32)
        elif CV_IMG_TYPE == cv2.IMREAD_COLOR:
            output_img = np.zeros((IMG_DIM[0],IMG_DIM[1]*3,IMG_DIM[2]), \
                                  dtype=np.float32)
        else:
            print('Incorrect image type')
            return
        
        cnt = 0
        for d in data_cur:
            input_imgs = d[0]
            label_imgs = d[1]
            pred_imgs = sess.run(self.G_z, \
                                feed_dict={self.Z: input_imgs})
            
            for i in range(self.batch_size):
                input_img = input_imgs[i]
                label_img = label_imgs[i]
                
                pred_img = pred_imgs[i]
                pred_img = pred_img - pred_img.min()
                pred_img = pred_img/pred_img.max()
                
                if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
                    output_img[:,0:IMG_DIM[1]] = input_img[:,:,0]
                    output_img[:,(IMG_DIM[1]):(2*IMG_DIM[1])] = \
                                                            pred_img[:,:,0]
                    output_img[:,(2*IMG_DIM[1]):(3*IMG_DIM[1])] = \
                                                            label_img[:,:,0]
                elif CV_IMG_TYPE == cv2.IMREAD_COLOR:
                    output_img[:,0:IMG_DIM[1],:] = input_img
                    output_img[:,(IMG_DIM[1]):(2*IMG_DIM[1]),:] = pred_img
                    output_img[:,(2*IMG_DIM[1]):(3*IMG_DIM[1]),:] = label_img
                else:
                    return
                
                output_img = output_img * 255.0
                
                output_img_path = join(output_path, str(cnt)+'.jpg')
                cv2.imwrite(output_img_path, output_img)
                cnt += 1


    def discriminator(self, x, scope):
        with tf.variable_scope(scope):
            #Convolution Layer 1
            D_conv1 = tf.layers.conv2d(inputs=x, filters=16, \
                                       kernel_size=[5,5], padding='same', \
                                       activation=tf.nn.relu, name='D_conv1')
            D_pool1 = tf.layers.max_pooling2d(inputs=D_conv1, \
                                              pool_size=[2,2], strides=2, \
                                              name='D_pool1')
            
            #Convolution Layer 2
            D_conv2 = tf.layers.conv2d(inputs=D_pool1, filters=32, \
                                       kernel_size=[5,5], padding='same', \
                                       activation=tf.nn.relu, name='D_conv2')
            D_pool2 = tf.layers.max_pooling2d(inputs=D_conv2, \
                                              pool_size=[2,2], strides=2, \
                                              name='D_pool2')
            
            #Flattening
            D_pool2_flat = tf.reshape(D_pool2, [-1, \
                           int(((IMG_DIM[0]/2.0)/2.0)*((IMG_DIM[1]/2.0)/2.0)\
                               *32)])
            
            #Output
            D_dense = tf.layers.dense(inputs=D_pool2_flat, units=1024, \
                                      activation=tf.nn.relu, name='D_dense')
            D_dropout = tf.layers.dropout(inputs=D_dense, rate=0.4, \
                                          training= True, name='D_dropout')
            D_logits = tf.layers.dense(inputs=D_dropout, units=1, \
                                       name='D_logits')
            D_probs = tf.nn.softmax(D_logits, name="D_probs")
            
            return D_probs, D_logits
        
    
    def encoder(self, z):
        #Convolution Layer 1
        G_conv1 = tf.layers.conv2d(inputs=z, filters=32, kernel_size=[5,5], \
                                   padding='same', activation=tf.nn.relu, \
                                   name='G_conv1')
        G_pool1 = tf.layers.max_pooling2d(inputs=G_conv1, pool_size=[2,2], \
                                          strides=2, name='G_pool1')
        
        #Convolution Layer 2
        G_conv2 = tf.layers.conv2d(inputs=G_pool1, filters=64, \
                                        kernel_size=[5,5], padding='same', \
                                        activation=tf.nn.relu, name='G_conv2')
        G_pool2 = tf.layers.max_pooling2d(inputs=G_conv2, \
                                          pool_size=[2,2], strides=2, \
                                          name='G_pool2')
        
        #Flattening
        flat_sz = int(((IMG_DIM[0]/2.0)/2.0)*((IMG_DIM[1]/2.0)/2.0)*64)
        G_pool2_flat = tf.reshape(G_pool2, [-1, flat_sz])
        
        #Output
        G_mean = tf.layers.dense(inputs=G_pool2_flat, units=64, \
                                 activation=tf.nn.relu, name='G_mean')
        G_stddev = tf.layers.dense(inputs=G_pool2_flat, units=64, 
                                   activation=tf.nn.relu, name='G_stddev')
        
        return G_mean, G_stddev
    
    
    def decoder(self, q):
        #Reshapping
        flat_sz = int(((IMG_DIM[0]/2.0)/2.0)*((IMG_DIM[1]/2.0)/2.0)*64)
        G_z_develop = tf.layers.dense(inputs=q, units=flat_sz, \
                                      activation=tf.nn.relu, \
                                      name='G_z_matrix')
        
        inp_dim = [-1]
        [inp_dim.append(i) for i in IMG_DIM]
        G_z_matrix = tf.reshape(G_z_develop, [self.batch_size, \
                                              int((IMG_DIM[0]/2.0)/2.0), \
                                              int((IMG_DIM[1]/2.0)/2.0), 64])
        G_z_matrix = tf.nn.relu(G_z_matrix)
        
        #Deconvolution Layer 1
        G_filter1 = tf.get_variable('G_filter1', [5, 5, 32, 64], \
                                    initializer=tf.random_normal_initializer(\
                                                stddev=0.02))
        G_deconv1 = tf.nn.conv2d_transpose(G_z_matrix, G_filter1, \
                                           output_shape=[self.batch_size,\
                                                         int(IMG_DIM[0]/2.0),\
                                                         int(IMG_DIM[1]/2.0),\
                                                         32],\
                                           strides = [1,2,2,1], \
                                           name='G_deconv1')
        G_deconv1 = tf.nn.relu(G_deconv1)
        
        #Deconvolution Layer 2
        G_filter2 = tf.get_variable('G_filter2', [5, 5, IMG_DIM[2], 32], \
                                    initializer=tf.random_normal_initializer(\
                                                stddev=0.02))
        G_deconv2 = tf.nn.conv2d_transpose(G_deconv1, G_filter2, \
                                           output_shape=[self.batch_size,\
                                                         IMG_DIM[0],\
                                                         IMG_DIM[1],\
                                                         IMG_DIM[2]],\
                                           strides = [1,2,2,1], \
                                           name='G_deconv2')
        G_deconv2 = tf.nn.sigmoid(G_deconv2)
        
        return G_deconv2
    
    
    def generator(self, z):
        z_mean, z_stddev = self.encoder(z)
        samples = tf.random_normal(shape=[self.batch_size, 64], mean=0.0, \
                                   stddev=1.0, dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)
        generated_images = self.decoder(guessed_z)
        
        return generated_images, z_mean, z_stddev
        
        
    def xavier_init(self, size):
        std_dev = 1.0/tf.sqrt(size[0]/2.0)
        return tf.random_normal(shape=size, stddev=std_dev)
        
    
    def get_data(self, batch_size, file_name):
        
        f_name = join(join(getcwd(), 'img_data'), file_name)
        f = open(f_name, 'r+')
        
        f_rand = f.readlines()
        shuffle(f_rand)
        
        count = 0
        
        inp_dim = [batch_size]
        [inp_dim.append(i) for i in IMG_DIM]
        inp_dim = tuple(inp_dim)
        
        input_imgs = np.empty(inp_dim)
        output_imgs = np.empty(inp_dim)
            
        for line in f_rand:
            input_img, output_img = line.split(',')
            output_img = output_img.splitlines()[0]
            
            input_img_arr = cv2.imread(input_img, CV_IMG_TYPE)
            input_img_arr = np.reshape(input_img_arr, IMG_DIM)
            input_img_arr = input_img_arr/255.0
            output_img_arr = cv2.imread(output_img, CV_IMG_TYPE)
            output_img_arr = np.reshape(output_img_arr, IMG_DIM)
            output_img_arr = output_img_arr/255.0
            
            input_imgs[count%batch_size] = input_img_arr
            output_imgs[count%batch_size] = output_img_arr
            
            count += 1
            if count%batch_size == 0:
                yield input_imgs, output_imgs
        
        yield input_imgs, output_imgs
    
    
    
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    test_gan = GAN()
    test_gan.train(epochs=20, tf_session=sess)
    test_gan.test(sess)
    
    sess.close()


if __name__ == '__main__':
    main()
    

#    def test(self, tf_session):
#        sess = tf_session
#        
#        f_name = join(join(getcwd(), 'img_data'), 'testing_file.txt')
#        f = open(f_name, 'r+')
#        
#        output_path = join(join(getcwd(), 'img_data'), 'output_img_data')
#        
#        if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
#            output_img = np.zeros((IMG_DIM[0],IMG_DIM[1]*3), \
#                                  dtype=np.float32)
#        elif CV_IMG_TYPE == cv2.IMREAD_COLOR:
#            output_img = np.zeros((IMG_DIM[0],IMG_DIM[1]*3,IMG_DIM[2]), \
#                                  dtype=np.float32)
#        else:
#            print('Incorrect image type')
#            return
#        
#        for line in f:
#            input_img_path, label_img_path = line.split(',')
#            label_img_path = label_img_path.splitlines()[0]
#            output_img_path = join(output_path, basename(label_img_path))
#            
#            input_img = cv2.imread(input_img_path, CV_IMG_TYPE)
#            input_img = np.reshape(input_img, IMG_DIM)
#            input_img = input_img/255.0
#            
#            pred_img = sess.run(self.G_z, \
#                                feed_dict={self.Z: np.array([input_img])})
#            pred_img = pred_img - pred_img.min()
#            pred_img = pred_img/max(1.0, pred_img.max())
#            print(pred_img.shape)
#            
#            label_img = cv2.imread(label_img_path, CV_IMG_TYPE)
#            label_img = np.reshape(label_img, IMG_DIM)
#            label_img = label_img/255.0
#            
#            if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
#                output_img[:,0:IMG_DIM[1]] = input_img
#                output_img[:,(IMG_DIM[1]):(2*IMG_DIM[1])] = pred_img
#                output_img[:,(2*IMG_DIM[1]):(3*IMG_DIM[1])] = label_img
#            elif CV_IMG_TYPE == cv2.IMREAD_COLOR:
#                output_img[:,0:IMG_DIM[1],:] = input_img
#                output_img[:,(IMG_DIM[1]):(2*IMG_DIM[1]),:] = pred_img
#                output_img[:,(2*IMG_DIM[1]):(3*IMG_DIM[1]),:] = label_img
#            else:
#                return
#            
#            output_img = output_img * 255.0
#            
#            cv2.imwrite(output_img_path, output_img)
