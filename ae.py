# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf


class TFAutoEncoder:
    """auto-encoder model."""

    def __init__(self, hidden_dim):
        """initialization"""
        self._hidden_dim = hidden_dim

    def fit(self, input, iter=1000, graph=None, session=None, noising=False):
        """fit this model"""
        self.build_graph(input, noising, graph, session)

        with self.graph.as_default():
            self.init = tf.initialize_all_variables()
            sess = self.sess
            sess.run(self.init)
            self.init_noise = tf.initialize_variables([self.noise])
            for step in xrange(iter):
                sess.run(self.init_noise)
                sess.run(self.training_op, feed_dict=self.feed_dict)
                if step % 100 == 0:
                    print(sess.run(self.loss, feed_dict=self.feed_dict))
                    #print(sess.run(self.noise))
                    #print(sess.run(self.mask_noise))
                    summary_str = sess.run(self.summary_op, feed_dict=self.feed_dict)
                    self.summary_writer.add_summary(summary_str, step)


    def encode(self, input_placeholder):
        """encode input"""
        with tf.name_scope('encode') as scope:
            self.noise = tf.Variable(tf.random_normal([self._input_dim], mean=0.0,
                                    stddev=1./10.), trainable=False, name='noise')
            self.W = tf.Variable(tf.random_normal([self._input_dim, self._hidden_dim]),
                                                    name='weight')
            self.b = tf.Variable(tf.random_normal([self._hidden_dim]), name='bias')
            encoded = tf.sigmoid(tf.matmul(input_placeholder + self.noise *
                                                self.mask_noise, self.W) + self.b)
        return encoded

    def reconstruct(self, encoded):
        """reconstruct encoded data"""
        with tf.name_scope('reconstruct') as scope:
            self.Wt = tf.transpose(self.W)
            self.c = tf.Variable(tf.random_normal([self._input_dim]), name='bias')
            reconstructed = tf.sigmoid(tf.matmul(encoded, self.Wt) + self.c)
        return reconstructed

    def loss(self, reconstructed, supervisor_placeholder):
        with tf.name_scope('loss') as scope:
            sq_err = tf.reduce_sum(tf.square(supervisor_placeholder - reconstructed))
            tf.scalar_summary('sq_err', sq_err)
        return sq_err

    def training(self, loss):
        with tf.name_scope('training') as scope:
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        return train_step

    def build_graph(self, input, noising, graph, session):
        """build a graph"""
        self.input = input
        self._input_dim = len(input[0])
        self.noising = noising

        if graph == None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        with self.graph.as_default():
            if not self.noising:
                self.mask_noise = tf.Variable(tf.zeros([self._input_dim]),
                                    trainable=False, name='mask')
            elif self.noising:
                self.mask_noise = tf.Variable(tf.ones([self._input_dim]),
                                    trainable=False, name='mask')

            self.input_placeholder = tf.placeholder("float", [None, self._input_dim],
                                            name="input_placeholder")
            self.supervisor_placeholder = tf.placeholder("float", [None, self._input_dim],
                                            name="supervisor_placeholder")

            self.feed_dict = {self.input_placeholder: self.input,
                                self.supervisor_placeholder: self.input}

            self.encoded = self.encode(self.input_placeholder)
            self.reconstructed = self.reconstruct(self.encoded)
            self.loss = self.loss(self.reconstructed, self.supervisor_placeholder)
            self.training_op = self.training(self.loss)

            self.summary_op = tf.merge_all_summaries()

            if session == None:
                self.sess = tf.Session()
            else:
                self.sess = session
            sess = self.sess
            self.summary_writer = tf.train.SummaryWriter('log', graph_def=sess.graph_def)

            self.init = tf.initialize_all_variables()
            sess = self.sess
            sess.run(self.init)

    def get_encoded(self, input):
        input = np.asarray(input)
        #with self.graph.as_default():
        sess = self.sess
        w = sess.run(self.W)
        b = sess.run(self.b)
        self.encoded = sess.run(tf.sigmoid(input.dot(w) + b))
        return self.encoded

    def get_reconstructed(self, encoded):
        #with self.graph.as_default():
        sess = self.sess
        wt = sess.run(self.Wt)
        c = sess.run(self.c)
        self.reconstructed = sess.run(tf.sigmoid(encoded.dot(wt) + c))
        return self.reconstructed

    def get_autoencoded(self, input):
        encoded = self.get_encoded(input)
        self.reconstructed = self.get_reconstructed(encoded)
        return self.reconstructed

    def get_W(self):
        sess = self.sess
        return sess.run(self.W)
