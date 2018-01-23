# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:05:00 2018

@author: Syed Mohsin Bukhari
"""

import tensorflow as tf
import numpy as np
import cv2
from os.path import join, isfile, isdir
from os import getcwd, unlink, listdir, mkdir
from time import localtime, strftime
from utils.layers import conv2d, conv2d_t, dense
from utils.input_pipeline import get_img_data

CV_IMG_TYPE = cv2.IMREAD_COLOR
#CV_IMG_TYPE = cv2.IMREAD_GRAYSCALE

IMG_DIM = [128, 128]
IMG_CHN = (2*CV_IMG_TYPE) + 1
IMG_SZ = IMG_DIM[0]*IMG_DIM[1]*IMG_CHN

class GAN:

    def __init__(self):
        #Helper Attributes
        self.training_start_time = 0

        #Defining Placeholders
        inp_dim = [None, IMG_CHN]
        [inp_dim.append(i) for i in IMG_DIM]
        self.X = tf.placeholder(dtype=tf.float32, shape=inp_dim)
        self.Z = tf.placeholder(dtype=tf.float32, shape=inp_dim)
        self.batch_size = 128

        #Get Generator and Discriminator Outputs
        self.G_z, self.G_z_feats = self.generator(self.Z)
        self.D_x, self.D_x_logit = self.discriminator(self.X, 'real')
        self.D_z, self.D_z_logit = self.discriminator(self.G_z, 'fake')

        #Defining Loss Functions
        generated_flat = tf.reshape(self.G_z, [self.batch_size, -1])
        original_flat = tf.reshape(self.X, [self.batch_size, -1])
        self.generation_loss = -tf.reduce_sum(original_flat * tf.log(1e-8 + \
            generated_flat) + (1-original_flat) * tf.log(1e-8 + 1 - \
            generated_flat), 1)
        self.cost = tf.reduce_mean(self.generation_loss)

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
        self.D_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).\
                                                    minimize(self.D_loss, \
                                                 var_list = self.D_vars)
        self.G_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).\
                                                    minimize(self.G_loss, \
                                                 var_list = self.G_vars)
        self.AE_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).\
                                minimize(self.cost, var_list = self.G_vars)


    def discriminator(self, x, scope, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        with tf.variable_scope(scope):
            #Convolutional Layers
            D_conv1 = conv2d(x, 16, 3, 'D_conv1', strides=2)
            D_conv2 = conv2d(D_conv1, 32, 3, 'D_conv2', strides=2)
            D_conv3 = conv2d(D_conv2, 64, 3, 'D_conv3', strides=2)
            D_conv4 = conv2d(D_conv3, 128, 3, 'D_conv4', strides=2)
            D_conv5 = conv2d(D_conv4, 256, 3, 'D_conv5')

            #Flattening
            D_pool5_flat = tf.layers.flatten(D_conv5)

            #Output
            D_dense = dense(inputs=D_pool5_flat, units=2048, name='D_dense')
            D_logits = dense(inputs=D_dense, units=1, name='D_logits',
                             activation=None)
            D_probs = tf.nn.softmax(D_logits, name="D_probs")

            return D_probs, D_logits


    def generator(self, z):
        ## Encoder
        #Convolution Layers
        G_conv1 = conv2d(z, 16, 3, 'G_conv1', strides=2)
        G_conv2 = conv2d(G_conv1, 32, 3, 'G_conv2', strides=2)
        G_conv3 = conv2d(G_conv2, 64, 3, 'G_conv3', strides=2)
        G_conv4 = conv2d(G_conv3, 128, 3, 'G_conv4', strides=2)
        G_conv5 = conv2d(G_conv4, 256, 3, 'G_conv5')

        #Flattening
        G_pool5_flat = tf.layers.flatten(G_conv5)

        #Output
        G_features = dense(inputs=G_pool5_flat, units=512, name='G_features')

        ## Decoder
        #Reshapping
        flat_sz = int(IMG_DIM[0]*IMG_DIM[1]*256*(2**-8))
        G_z_dev = dense(inputs=G_features, units=flat_sz, name='G_z_dev')

        G_z_mat = tf.reshape(G_z_dev, tf.shape(G_conv5))
        G_z_matrix = tf.nn.relu(G_z_mat, name='G_z_matrix')

        #Transpose Convolutional Layers
        G_conv_t_1 = conv2d_t(G_z_matrix, 256, 3, 'G_conv_t_1')
        G_conv_t_2 = conv2d_t(G_conv_t_1, 128, 3, 'G_conv_t_2')
        G_conv_t_3 = conv2d_t(G_conv_t_2, 64, 3, 'G_conv_t_3')
        G_conv_t_4 = conv2d_t(G_conv_t_3, 32, 3, 'G_conv_t_4')
        G_conv_t_5 = conv2d_t(G_conv_t_4, IMG_CHN, 3, 'G_conv_t_5', strides=1,
                              activation=tf.nn.tanh)

        G_pic = tf.divide(tf.add(G_conv_t_5, 1.0), 2.0)

        return G_pic, G_features


    def train(self, epochs, tf_session):
        self.training_start_time = strftime("%d%m%Y%H%M%S", localtime())

        sess = tf_session
        sess.run(tf.global_variables_initializer())

        batch_size = self.batch_size

        for i in range(epochs):

            data_cur = get_img_data(batch_size, 'training_file.txt',
                                    CV_IMG_TYPE, IMG_CHN, IMG_DIM)

            for d in data_cur:
#                for k in range(2):
#                    _, D_loss_cur = sess.run([self.D_optimizer,
#                                              self.D_loss], \
#                                              feed_dict={self.X: d[1], \
#                                                         self.Z: d[0]})
#                for k in range(1):
#                    _, G_loss_cur = sess.run([self.G_optimizer, \
#                                              self.G_loss], \
#                                              feed_dict={self.Z: d[0]})
                for k in range(1):
                    _, generation_loss =  sess.run([self.AE_optimizer, \
                                          self.cost], \
                                          feed_dict={self.X: d[1], \
                                                     self.Z: d[0]})

            if i%1==0:
                print('Epoch: {}'.format(i))
#                print('D loss: {0}'.format(D_loss_cur))
#                print('G_loss: {0}'.format(G_loss_cur))
                print('mean_generation_loss: {0}'.format(np.mean(
                        generation_loss)))
                print()

            self.test(sess, epoch='epoch_'+str(i))


    def test(self, tf_session, epoch='final'):
        sess = tf_session
        data_cur = get_img_data(self.batch_size, 'testing_file.txt',
                                CV_IMG_TYPE, IMG_CHN, IMG_DIM)

        #construct output path
        output_path = join(join(getcwd(), 'img_data'), 'output_img_data')
        output_path = join(output_path, self.training_start_time)
        if not isdir(output_path):
            mkdir(output_path)
        output_path = join(output_path, epoch)
        if not isdir(output_path):
            mkdir(output_path)

        #delete all previous files
        for the_file in listdir(output_path):
            file_path = join(output_path, the_file)
            try:
                if isfile(file_path):
                    unlink(file_path)
            except Exception as e:
                print(e)

        output_img = np.zeros((IMG_CHN, IMG_DIM[0],IMG_DIM[1]*3),
                              dtype=np.float32)

        cnt = 0
        for d in data_cur:
            input_imgs = d[0]
            label_imgs = d[1]
            pred_imgs = sess.run(self.G_z, feed_dict={self.Z: input_imgs})

            for i in range(self.batch_size):
                input_img = input_imgs[i]
                label_img = label_imgs[i]

                pred_img = pred_imgs[i]
                pred_img = pred_img - pred_img.min()
                pred_img = pred_img/pred_img.max()

                output_img[:,:,0:IMG_DIM[1]] = input_img
                output_img[:,:,(IMG_DIM[1]):(2*IMG_DIM[1])] = pred_img
                output_img[:,:,(2*IMG_DIM[1]):(3*IMG_DIM[1])] = label_img

                output_img = output_img * 255.0

                if CV_IMG_TYPE == cv2.IMREAD_GRAYSCALE:
                    output_img_final = output_img[0,:,:]
                else:
                    output_img_final = np.swapaxes(output_img, 0, 2)
                    output_img_final = np.swapaxes(output_img_final, 0, 1)

                output_img_path = join(output_path, str(cnt)+'.jpg')
                cv2.imwrite(output_img_path, output_img_final)
                cnt += 1



def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
      test_gan = GAN()
      test_gan.train(epochs=50, tf_session=sess)
      test_gan.test(sess)
      sess.close()


if __name__ == '__main__':
    main()
