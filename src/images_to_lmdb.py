#!/usr/bin/env python
import sys
caffe_root = '/home/jiangliang/code/future/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from PIL import Image
import numpy as np
import lmdb
from util import *

data_path = './data/Flickr8k/'
train_file = 'split/train_list'
val_file = 'split/val_list'
test_file = 'split/test_list'
mean_file = 'mean_color'
image_path = data_path + 'images/'
image_size = np.array([227, 227])

f = open(data_path + train_file)
train_list = f.readlines()
f = open(data_path + val_file)
val_list = f.readlines()
f = open(data_path + test_file)
test_list = f.readlines()

def to_lmdb(y_lmdb, uv_lmdb, file_list):
    num_images = 0
    mean_color = np.zeros((3))
    with y_lmdb.begin(write = True) as y_image_txn:
        with uv_lmdb.begin(write = True) as uv_image_txn:
            for i in range(len(file_list)):
                print i
                im_name = file_list[i][:-1] 
                im = Image.open(image_path + im_name) 
                im = im.resize(image_size)
                im = np.array(im)                  
                im_yuv = rgb2yuv(im)
                im_yuv = im_yuv.transpose((2, 0, 1))
                for j in range(3):
                    mean_color[j] += im_yuv[j, :, :].mean()
                num_images += 1
                y = im_yuv[0, :, :]
                uv = im_yuv[1 : 3, :, :]
                
                y_data = caffe.proto.caffe_pb2.Datum()
                y_data.channels = 1
                y_data.height, y_data.width = y.shape
                y_data.data = y.tostring()
                y_image_txn.put('{:0>12d}'.format(i), y_data.SerializeToString())
               
                uv_data = caffe.proto.caffe_pb2.Datum()
                uv_data.channels, uv_data.height, uv_data.width = uv.shape
                uv_data.data = uv.tostring()
                uv_image_txn.put('{:>12d}'.format(i), uv_data.SerializeToString())
        uv_lmdb.close()
    y_lmdb.close()
    mean_color  = np.double(mean_color) / num_images
    return num_images, mean_color

print "processing training set..."
train_y_lmdb_name = 'train_y.lmdb'
train_uv_lmdb_name = 'train_uv.lmdb'
train_y_lmdb = lmdb.open(data_path + train_y_lmdb_name, map_size = int(1e12))
train_uv_lmdb = lmdb.open(data_path + train_uv_lmdb_name, map_size = int(1e12))
[num_train, train_mean] = to_lmdb(train_y_lmdb, train_uv_lmdb, train_list)
print('num_train: {}, train_mean: {}'.format(num_train, train_mean))

print "processing validation set..."
val_y_lmdb_name = 'val_y.lmdb'
val_uv_lmdb_name = 'val_uv.lmdb'
val_y_lmdb = lmdb.open(data_path + val_y_lmdb_name, map_size = int(1e12))
val_uv_lmdb = lmdb.open(data_path + val_uv_lmdb_name, map_size = int(1e12))
[num_val, val_mean] = to_lmdb(val_y_lmdb, val_uv_lmdb, val_list)
print('num_val: {}, val_mean: {}'.format(num_val, val_mean))

print "processing test set..."
test_y_lmdb_name = 'test_y.lmdb'
test_uv_lmdb_name = 'test_uv.lmdb'
test_y_lmdb = lmdb.open(data_path + test_y_lmdb_name, map_size = int(1e12))
test_uv_lmdb = lmdb.open(data_path + test_uv_lmdb_name, map_size = int(1e12))
[num_test, test_mean] = to_lmdb(test_y_lmdb, test_uv_lmdb, test_list)
print('num_test: {}, test_mean: {}'.format(num_test, test_mean))

np.savez(data_path + mean_file, num_train = num_train, train_mean = train_mean,
        num_val = num_val, val_mean = val_mean, 
        num_test = num_test, test_mean = test_mean)
