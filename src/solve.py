#!/usr/bin/env python
import sys
caffe_root = '/usr/local/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

steps = 80000
caffe.set_mode_gpu()
caffe.set_device(0)
train_loss = np.zeros(steps)
solver = caffe.SGDSolver('/home/jiangliang/code/caffe_colorization/models/solver.prototxt')
for i in range(steps):
    solver.step(1) 
    train_loss[i] = solver.net.blobs['loss'].data
np.savetxt('/home/jiangliang/code/caffe_colorization/result/loss', train_loss)
