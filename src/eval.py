#!/usr/bin/env python
import sys
caffe_root = '/usr/local/caffe/'
project_root = '/home/jiangliang/code/caffe_colorization/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from PIL import Image
import numpy as np
from util import *
import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = "model to evaluate")
    return parser

parser = argparser()
args = parser.parse_args()
model = args.model
net = caffe.Net(project_root + 'models/train_val_fuse.prototxt',
                model, 
                caffe.TEST)

image_size = np.array([227, 227])
image_path = project_root + 'data/flowers/images/image_07863.jpg'
image = Image.open(image_path)
image = image.resize(image_size)

im = np.array(image)
yuv = rgb2yuv(im)
y = yuv[:, :, 0]
shape = y.shape
y = y[np.newaxis, :]
#print net.blobs['data'].data.shape
#net.blobs['data'].reshape(1, *y.shape)
#print net.blobs['data'].data.shape
net.blobs['data'].data[...] = y
#net.blobs['label'].reshape(1, 2, *shape)
#print "label shape"
#print net.blobs['label'].data.shape
label = yuv[:, :, 1 : 3]
label = label.transpose((2, 0, 1))
net.blobs['label'].data[...] = label

for layer, data in net.blobs.items():
    print layer + '\t' + str(data.data.shape)
#item = net.blobs.items()
net.forward()
#out = net.blobs['upscore'].data
#out = net.blobs['upscore'].data
out = net.blobs.items()[-2][1].data
print 'loss: \t' + str(net.blobs['loss'].data)
y = y.squeeze()
out_yuv = np.zeros((y.shape[0], y.shape[1], 3))
out_yuv[:, :, 0] = y
out = out.squeeze()
out_yuv[:, :, 1] = out[0, :, :]
out_yuv[:, :, 2] = out[1, :, :]
out_rgb = yuv2rgb(out_yuv)
or_rgb = yuv2rgb(yuv)

out_im = Image.fromarray(out_rgb)
or_im = Image.fromarray(or_rgb)

or_im.show()
out_im.show()
