# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf


class TFAutoEncoder:
    """auto-encoder model."""

    def __init__(self, hidden_dim):
        """initialization"""
        self._hidden_dim = hidden_dim

    def fit(self, iter=1000):
        """fit this model"""
        with self.graph.as_default():
            self.init = tf.initialize_all_variables()
            sess = self.sess
            sess.run(self.init)
            for step in xrange(iter):
                sess.run(self.training_op, feed_dict=self.feed_dict)
                if step % 100 == 0:
                    print(sess.run(self.loss, feed_dict=self.feed_dict))
                    print(sess.run(self.W))
                    summary_str = sess.run(self.summary_op, feed_dict=self.feed_dict)
                    self.summary_writer.add_summary(summary_str, step)


    def encode(self, input_placeholder):
        """encode input"""
        with tf.name_scope('encode') as scope:
            self.W = tf.Variable(tf.random_normal([self._input_dim, self._hidden_dim]), name='weight')
            self.b = tf.Variable(tf.random_normal([self._hidden_dim], name='bias'))
            encoded = tf.sigmoid(tf.matmul(input_placeholder, self.W) + self.b)
        return encoded

    def reconstruct(self, encoded):
        """reconstruct encoded data"""
        with tf.name_scope('reconstruct') as scope:
            self.Wt = tf.transpose(self.W)
            self.c = tf.Variable(tf.random_normal([self._input_dim], name='bias'))
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

    def build_graph(self, input, input_dim, graph=None):
        """build a graph"""
        self._input_dim = input_dim

        if graph == None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        with self.graph.as_default():
            self.input_placeholder = tf.placeholder("float", [None, input_dim],
                                            name="input_placeholder")
            self.supervisor_placeholder = tf.placeholder("float", [None, input_dim],
                                            name="supervisor_placeholder")

            self.feed_dict = {self.input_placeholder: input,
                                self.supervisor_placeholder: input}

            self.encoded = self.encode(self.input_placeholder)
            self.reconstructed = self.reconstruct(self.encoded)
            self.loss = self.loss(self.reconstructed, self.supervisor_placeholder)
            self.training_op = self.training(self.loss)

            self.summary_op = tf.merge_all_summaries()

            self.sess = tf.Session()
            sess = self.sess
            self.summary_writer = tf.train.SummaryWriter('log', graph_def=sess.graph_def)

    def get_encoded(self, input):
        with self.graph.as_default():
            sess = self.sess
            w = sess.run(self.W)
            b = sess.run(self.b)
            encoded = sess.run(tf.sigmoid(input.dot(w) + b))
            return encoded

    def get_reconstructed(self, encoded):
        with self.graph.as_default():
            sess = self.sess
            wt = sess.run(self.Wt)
            c = sess.run(self.c)
            decoded = sess.run(tf.sigmoid(encoded.dot(wt) + c))
            return decoded
