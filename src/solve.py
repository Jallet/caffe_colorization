#!/usr/bin/env python
import sys
import argparse
caffe_root = '/home/jiangliang/code/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', action = 'store', dest = 'iters', default = 10000, type = int, help = "training iterations")
    return parser

parser = argparser()
args = parser.parse_args()
iters = args.iters
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.NesterovSolver('/home/jiangliang/code/caffe_colorization/models/solver.prototxt')
solver.step(iters)
