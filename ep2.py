# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
import os

from ae import TFAutoEncoder
from sdae import TFStackedDenoisingAutoEncoder

os.system('rm -rf log')

input = [
  [1., 0., 0., 0., 0.],
  [0., 1., 0., 0., 0.],
  [0., 0., 1., 0., 0.],
  [0., 0., 0., 1., 0.],
  [0., 0., 0., 0., 1.],
]

sdae = TFStackedDenoisingAutoEncoder(units_list=[3,2,3])
#sdae.build_graph(input=input)
sdae.fit(input=input, iter=10000)
