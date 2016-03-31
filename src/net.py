#!/usr/bin/env python
caffe_root = '/home/jiangliang/code/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

net = caffe.Net('models/train_val_fuse.prototxt', caffe.TRAIN)
print 'shape of data:'
for k, v in net.blobs.items():
    print k, v.data.shape

print 'shape of parmas:'
for k, v in net.params.items():
    print k, v[0].data.shape
