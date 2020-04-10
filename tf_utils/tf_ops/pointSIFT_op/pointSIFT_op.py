"""
Author: Jiang Mingyang
email: jmydurant@sjtu.edu.cn
DFCN module op, do not modify it !!!
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

DFCN_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_DFCN_so.so'))

def DFCN_select(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius: float
    :return: (b, n, 8) int
    """
    idx = DFCN_module.cube_select(xyz, radius)
    return idx


ops.NoGradient('CubeSelect')

def DFCN_select_two(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 16) int
    """
    idx = DFCN_module.cube_select_two(xyz, radius)
    return idx


ops.NoGradient('CubeSelectTwo')

def DFCN_select_four(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 32) int
    """
    idx = DFCN_module.cube_select_four(xyz, radius)
    return idx


ops.NoGradient('CubeSelectFour')


def DFCN_select_eight(xyz, radius):
    """
    :param xyz: (b, n, 3) float
    :param radius:  float
    :return: idx: (b, n, 32) int
    """
    idx = DFCN_module.cube_select_eight(xyz, radius)
    return idx


ops.NoGradient('CubeSelectEight')