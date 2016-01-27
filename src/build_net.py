#!/usr/bin/env python
import caffe
from caffe import layers as L
from caffe import params as P

import sys

def colorization_net(train_data, test_data, batch_size):
    net = caffe.NetSpec()
    net.data = L.Data(batch_size = batch_size,
                    backend = P.Data.LMDB, source = train_data, phase = caffe.TRAIN)
    net.conv1 = L.Convolution(net.data, kernel_size = 5, 
                            num_output = 20, weight_filler = dict(type = 'xavier'))
    net.pool1 = L.Pooling(net.conv1, kernel_size = 2, 
                        stride = 2, pool = P.Pooling.MAX)
    net.ip1 = L.InnerProduct(net.pool1, num_output = 10, 
            weight_filler = dict(type = 'xavier'))
    net.loss = L.SoftmaxWithLoss(net.ip1, net.data)
    return net.to_proto()

def main(argv):
    model_dir = 'model/'
    model_file = 'colorization_model.prototxt'
    train_data_path = 'data/train_x_lmdb'
    test_data_path = 'data/test_x_lmdb'
    batch_size = 64
    with open(model_dir + model_file, 'w') as f:
        f.write(str(colorization_net(train_data_path, test_data_path, batch_size)))

if __name__ == '__main__':
    main(sys.argv)
