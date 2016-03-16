#!/bin/sh
date >> result/time
/usr/local/caffe/build/tools/caffe train --solver=/home/jiangliang/code/caffe_colorization/models/solver.prototxt 
date >> result/time
