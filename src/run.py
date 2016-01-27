#!/usr/bin/env python
import caffe

model_dir = 'model/'
model_file = 'colorization_model.prototxt'
solver_file = 'solver.prototxt'

caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver(model_dir + solver_file)
a = [(k, v.data.shape) for k, v in solver.net.blobs.items()]
