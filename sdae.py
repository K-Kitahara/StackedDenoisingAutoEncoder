# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from ae import TFAutoEncoder

class TFStackedDenoisingAutoEncoder:
    """stacked denoising auto-encoder"""

    def __init__(self, units_list):
        """initialization"""
        self._units_list = units_list

        self.aes = []
        for i in xrange(len(self._units_list)):
            ae = TFAutoEncoder(self._units_list[i])
            self.aes.append(TFAutoEncoder(self._units_list[i]))

    def build_graph(self, input):
            for i in xrange(len(self._units_list)):
                name_scope = 'AutoEncoder' + str(i)
                with tf.name_scope(name_scope) as scope:
                    if i == 0:
                        self.aes[i].build_graph(input=self.input,
                                            session=self.sess, graph=self.graph)
                    else:
                        self.aes[i].build_graph(input=self.aes[i-1].get_encoded(self.aes[i-1].input),
                                            session=self.sess, graph=self.graph)

            self.summary_op = tf.merge_all_summaries()

            sess = self.sess
            self.summary_writer = tf.train.SummaryWriter('log', graph_def=sess.graph_def)

            self.init = tf.initialize_all_variables()
            sess = self.sess
            sess.run(self.init)

    def fit(self, input, iter=1000, graph=None, session=None):
        """fit this model"""
        self.input = input
        self._input_dim = len(input[0])

        if graph == None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        with self.graph.as_default():
            if session == None:
                self.sess = tf.Session()
            else:
                self.sess = session

        with self.graph.as_default():
            for i in xrange(len(self._units_list)):
                if i == 0:
                    self.aes[i].fit(iter=iter, input=self.input)#, session=self.sess, graph=self.graph)
                    print(self.aes[i].get_encoded(self.aes[i].input))
                else:
                    self.aes[i].fit(iter=iter, input=self.aes[i-1].encoded)#, session=self.sess, graph=self.graph)
                    self.aes[i].get_encoded(self.aes[i].input)
