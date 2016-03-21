#!/usr/bin/env python
import sys
caffe_root = '/usr/local/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
from util import yuv2rgb
from PIL import Image
y_lmdb_file = '/home/jiangliang/code/caffe_colorization/data/flowers/train_y.lmdb'
uv_lmdb_file = '/home/jiangliang/code/caffe_colorization/data/flowers/train_uv.lmdb'
y_lmdb = lmdb.open(y_lmdb_file, map_size = int(1e12))
uv_lmdb = lmdb.open(uv_lmdb_file, map_size = int(1e12))
y_txn = y_lmdb.begin()
y_cursor = y_txn.cursor()
y_datum = caffe_pb2.Datum()

uv_txn = uv_lmdb.begin()
uv_cursor = uv_txn.cursor()
uv_datum = caffe_pb2.Datum()

for key, value in y_cursor:
    print key
    y_datum.ParseFromString(value)
    data = caffe.io.datum_to_array(y_datum)
    for key_l, value_l in uv_cursor:
        uv_datum.ParseFromString(value_l)
        label = caffe.io.datum_to_array(uv_datum)
        yuv = np.zeros((3, 227, 227))
        yuv[0, :, :] = data
        yuv[1 : 3, :, :] = label
        yuv = yuv.transpose((1, 2, 0))
        rgb = yuv2rgb(yuv)
        im = Image.fromarray(rgb)
        im.show()
        print "hello"
