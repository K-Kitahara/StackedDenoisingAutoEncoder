# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf

from ae import TFAutoEncoder

import os

os.system('rm -rf log')

input = [
  [1., 0., 0., 0., 0.],
  [0., 1., 0., 0., 0.],
  [0., 0., 1., 0., 0.],
  [0., 0., 0., 1., 0.],
  [0., 0., 0., 0., 1.],
]

input = np.asarray(input)

sess = tf.Session()
graph = tf.Graph()

ae = TFAutoEncoder(hidden_dim=2)
ae.fit(input=input, noising=False, iter=10000)
