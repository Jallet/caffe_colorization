#!/usr/bin/env python
import sys
caffe_root = '/home/jiangliang/code/caffe/'
project_root = '/home/jiangliang/code/caffe_colorization/'
test_split = project_root + 'data/flowers/split/test_list'
image_path = project_root + 'data/flowers/images/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from PIL import Image
import numpy as np
from util import *
import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help = "model to evaluate")
    parser.add_argument('--num', help = "number of images to test",
            action = 'store', dest = 'num', default = '2', type = int)
    return parser


def test(net, path):
    image_size = np.array([227, 227])
    #image_path = project_root + 'data/flowers/images/image_07863.jpg'
    image = Image.open(path)
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
    
    #or_im = Image.fromarray(or_rgb)
    #gray_im = or_im.convert('L')
    #or_gray = np.array(gray_im)
    #print or_gray.shape
    
    concat_im = np.concatenate((or_rgb, out_rgb), 1)
    print("shape of concat_im: {}".format(concat_im.shape)) 
    concat_image = Image.fromarray(concat_im) 
    concat_image.show()
    
def main():
    parser = argparser()
    args = parser.parse_args()
    model = args.model
    num = args.num

    net = caffe.Net(project_root + 'models/train_val_fuse.prototxt',
                model, 
                caffe.TEST)
    f = open(test_split)         
    test_list = f.readlines()
    for i in range(num):
        test(net, image_path + test_list[i][:-1])
        
if __name__ == '__main__':
    main()
