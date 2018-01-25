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
from time import localtime, strftime, time
from utils.layers import conv, conv_t, dense, bn
from utils.activations import lrelu, sigmoid, relu#, tanh
from utils.input_pipeline import get_img_data

CV_IMG_TYPE = cv2.IMREAD_COLOR
#CV_IMG_TYPE = cv2.IMREAD_GRAYSCALE

IMG_DIM = [128, 128]
IMG_CHN = (2*CV_IMG_TYPE) + 1
IMG_SZ = IMG_DIM[0]*IMG_DIM[1]*IMG_CHN

class GAN:

    def __init__(self, batch_size = 64):
        #Helper Attributes
        self.training_start_time = 0

        #Defining Placeholders
        inp_dim = [None, IMG_CHN]
        [inp_dim.append(i) for i in IMG_DIM]
        self.X = tf.placeholder(dtype=tf.float32, shape=inp_dim)
        self.Z = tf.placeholder(dtype=tf.float32, shape=inp_dim)
        self.batch_size = batch_size

        #Get Generator and Discriminator Outputs
        self.G_z, self.G_z_feats = self.generator(self.Z)
        self.D_x, self.D_x_ = self.discriminator(self.X)
        self.D_z, self.D_z_ = self.discriminator(self.G_z)

        #Defining Loss Functions (Alternative Loss Functions just before EOF)
        self.G_flat = tf.reshape(self.G_z, [self.batch_size, -1])
        self.O_flat = tf.reshape(self.X, [self.batch_size, -1])
        self.generation_loss = -tf.reduce_sum(self.O_flat * tf.log(1e-8 + \
            self.G_flat) + (1-self.O_flat) * tf.log(1e-8 + 1 - \
            self.G_flat), 1)
        self.cost = tf.reduce_mean(self.generation_loss)

        self.D_loss=-tf.reduce_mean(tf.log(self.D_x)+tf.log(1.0 - self.D_z))
        self.G_loss=-tf.reduce_mean(tf.log(self.D_z))

        #Prepare variable lists
        t_vars = tf.trainable_variables()
        self.D_vars = [var for var in t_vars if 'D_' in var.name]
        self.G_vars = [var for var in t_vars if 'G_' in var.name]

        #Define optimizers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).\
                                                        minimize(self.D_loss,
                                                     var_list = self.D_vars)

            self.G_optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).\
                                                        minimize(self.G_loss,
                                                     var_list = self.G_vars)

            self.AE_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).\
                                                        minimize(self.cost,
                                                     var_list = self.G_vars)


    def discriminator(self, x):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):

            #Convolutional Layers
            D_conv1 = bn(lrelu(conv(bn(x), 16, 3, 'D_conv1')))
            D_conv2 = bn(lrelu(conv(D_conv1, 32, 3, 'D_conv2')))
            D_conv3 = bn(lrelu(conv(D_conv2, 64, 3, 'D_conv3')))
            D_conv4 = bn(lrelu(conv(D_conv3, 128, 3, 'D_conv4')))
            D_conv5 = bn(lrelu(conv(D_conv4, 256, 3, 'D_conv5')))

            #Flattening
            D_pool5_flat = tf.layers.flatten(D_conv5)

            #Output
            D_dense = bn(lrelu(dense(D_pool5_flat, 2048, 'D_dense')))
            D_prob = sigmoid(dense(D_dense, 1, 'D_prob'))

            return D_prob, D_conv5


    def encoder(self, z):
        with tf.variable_scope("encoder"):
            ## Encoder
            #Convolution Layers
            G_conv1 = bn(lrelu(conv(bn(z), 64, 5, 'G_conv1')))
            G_conv2 = bn(lrelu(conv(G_conv1, 128, 5, 'G_conv2')))
            G_conv3 = bn(lrelu(conv(G_conv2, 256, 5, 'G_conv3')))
            G_conv4 = bn(lrelu(conv(G_conv3, 512, 5, 'G_conv4')))
            G_conv5 = bn(lrelu(conv(G_conv4, 1024, 5, 'G_conv5')))

            #Transpose Convolutional Layer for projection
            G_conv_t = conv_t(G_conv5, 1, 4, 'G_conv_t', padding='valid')

            #Reshaping
            G_features = tf.layers.flatten(lrelu(G_conv_t), 'G_features')

            return G_features


    def decoder(self, q):
        with tf.variable_scope("decoder"):
            ## Decoder
            #Reshapping
            G_z_matrix = tf.reshape(q, [-1, 1, 10, 10], name='G_z_matrix')

            #Convolutional Layer for projection
            G_conv = relu(conv(G_z_matrix, 1024, 3, 'G_conv', strides=3))

            #Transpose Convolutional Layers
            G_conv_t_1 = bn(relu(conv_t(G_conv, 512, 5, 'G_conv_t_1')))
            G_conv_t_2 = bn(relu(conv_t(G_conv_t_1, 256, 5, 'G_conv_t_2')))
            G_conv_t_3 = bn(relu(conv_t(G_conv_t_2, 128, 5, 'G_conv_t_3')))
            G_conv_t_4 = bn(relu(conv_t(G_conv_t_3, 64, 5, 'G_conv_t_4')))
            G_conv_t_5 = sigmoid(conv_t(G_conv_t_4, IMG_CHN, 5, 'G_conv_t_5'))

            return G_conv_t_5


    def generator(self, z):
        G_features = self.encoder(z)
        G_pic = self.decoder(G_features)

        return G_pic, G_features


    def train(self, epochs, sess):
        self.training_start_time = strftime("%d-%m-%Y_%H-%M-%S", localtime())

        sess.run(tf.global_variables_initializer())

        batch_size = self.batch_size
        epoch_verbosity = 1
        autoencoder_epochs = 10
        for i in range(epochs):
            if i%epoch_verbosity==0:
                print('Running epoch: {}'.format(i+1))

            start_time = time()

            if i<autoencoder_epochs:
                data_cur = get_img_data(batch_size, 'training_file.txt',
                                        CV_IMG_TYPE, IMG_CHN, IMG_DIM)
                for d in data_cur:
                    _, generation_loss = \
                                sess.run([self.AE_optimizer,
                                          self.cost],
                                         feed_dict={self.X: d[1],
                                                    self.Z: d[0]})

            else:
                data_cur = get_img_data(batch_size, 'training_file.txt',
                                        CV_IMG_TYPE, IMG_CHN, IMG_DIM)

                for d in data_cur:
                    _, D_loss, D_x = \
                                sess.run([self.D_optimizer,
                                          self.D_loss,
                                          self.D_x],
                                         feed_dict={self.X: d[1],
                                                    self.Z: d[0]})

                    for k in range(2):
                        _, G_loss, D_z = \
                                    sess.run([self.G_optimizer,
                                              self.G_loss,
                                              self.D_z],
                                             feed_dict={self.X: d[1],
                                                        self.Z: d[0]})

            time_taken = time()-start_time
            if i%epoch_verbosity==0:
                print('Time Taken: {}'.format(time_taken))
                if i>=autoencoder_epochs:
                    print('D_x: {0}'.format(np.mean(D_x)))
                    print('D_z: {0}'.format(np.mean(D_z)))
                    print('D_loss: {0}'.format(D_loss))
                    print('G_loss: {0}'.format(G_loss))
                else:
                    print('mean_generation_loss: {0}'.format(generation_loss))
                print()

            self.test(sess, epoch='epoch_'+str(i+1))


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

#                output_img_final = (output_img_final+1.0)/2.0

                output_img_path = join(output_path, str(cnt)+'.jpg')
                cv2.imwrite(output_img_path, output_img_final)
                cnt += 1



def main():
    with tf.Session() as sess:
      test_gan = GAN(batch_size=64)
      test_gan.train(epochs=100, sess=sess)
      sess.close()


if __name__ == '__main__':
    main()



#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth=True
#
#    with tf.Session(config=config) as sess:

#        # Alternative losses:
#        # -------------------
#        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
#                logits=self.D_x_logit, labels=tf.ones_like(self.D_x_logit)))
#        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
#                logits=self.D_z_logit, labels=tf.zeros_like(self.D_z_logit)))
#        self.D_loss = D_loss_real + D_loss_fake
#        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
#                logits=self.D_z_logit, labels=tf.ones_like(self.D_z_logit)))



#                for k in range(1):
#                    _, D_loss = sess.run([self.D_optimizer,
#                                              self.D_loss],
#                                             feed_dict={self.X: d[1],
#                                                        self.Z: d[0]})
#
#                for k in range(1):
#                    _, G_loss = sess.run([self.G_optimizer,
#                                              self.G_loss],
#                                             feed_dict={self.Z: d[0]})
#
#                for k in range(1):
#                    _, generation_loss =  sess.run([self.AE_optimizer, \
#                                          self.cost], \
#                                          feed_dict={self.X: d[1], \
#                                                     self.Z: d[0]})
#
#                D_x, D_z = sess.run([self.D_x, self.D_z],
#                                            feed_dict={self.X: d[1],
#                                                       self.Z: d[0]})