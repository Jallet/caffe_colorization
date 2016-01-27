#!/usr/bin/env python
#convert image data from ndarray to datum, generate lmdb
import caffe
import lmdb
import numpy as np

data_dir = 'data/'
data_file = 'mini_train.npz'
train_img_lmdb = 'train_lmdb'
test_img_lmdb = 'test_lmdb'

data = np.load(data_dir + data_file)
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

train_x_lmdb = 'train_x_lmdb'
test_x_lmdb = 'test_x_lmdb'
train_y_lmdb = 'train_y_lmdb'
test_y_lmdb = 'test_y_lmdb'

train_x_db = lmdb.open(data_dir + train_x_lmdb, map_size = int(1e12))
with train_x_db.begin(write = True) as txn:
    for i in range(train_x.shape[0]):
        img = train_x[i, :, :]
        img = np.expand_dims(img, axis = 0)
        img_data = caffe.io.array_to_datum(img)
        index = '{:0>10d}'.format(i)
        print(index)
        txn.put('{:0>10d}'.format(i), img_data.SerializeToString())
train_x_db.close()

train_y_db = lmdb.open(data_dir + train_y_lmdb, map_size = int(1e12))
with train_y_db.begin(write = True) as txn:
    for i in range(train_y.shape[0]):
        img = train_y[i, :, :]
        img = img.reshape([2, train_x.shape[1], train_x.shape[2]])
        #img = np.expand_dims(img, axis = 0)
        img_data = caffe.io.array_to_datum(img)
        index = '{:0>10d}'.format(i)
        print(index)
        txn.put('{:0>10d}'.format(i), img_data.SerializeToString())
train_y_db.close()

test_x_db = lmdb.open(data_dir + test_x_lmdb, map_size = int(1e12))
with test_x_db.begin(write = True) as txn:
    for i in range(test_x.shape[0]):
        img = test_x[i, :, :]
        img = np.expand_dims(img, axis = 0)
        img_data = caffe.io.array_to_datum(img)
        index = '{:0>10d}'.format(i)
        print(index)
        txn.put('{:0>10d}'.format(i), img_data.SerializeToString())
train_x_db.close()

test_y_db = lmdb.open(data_dir + test_y_lmdb, map_size = int(1e12))
with test_y_db.begin(write = True) as txn:
    for i in range(test_y.shape[0]):
        img = test_y[i, :, :]
        img = img.reshape([2, test_x.shape[1], test_x.shape[2]])
        #img = np.expand_dims(img, axis = 0)
        img_data = caffe.io.array_to_datum(img)
        index = '{:0>10d}'.format(i)
        print(index)
        txn.put('{:0>10d}'.format(i), img_data.SerializeToString())
test_y_db.close()
