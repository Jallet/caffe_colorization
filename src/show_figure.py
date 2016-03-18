#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import numpy as np
import argparse
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_loss_path")
    parser.add_argument("test_loss_path")
    return parser
parser = argparser()
args = parser.parse_args()
train_loss_path = args.train_loss_path
test_loss_path = args.test_loss_path
print train_loss_path

train_loss = np.loadtxt(train_loss_path)
test_loss = np.loadtxt(test_loss_path)
print train_loss
print train_loss.shape
print test_loss.shape
plt.figure(1)
plt.plot(train_loss, 'g-', label = "train_loss")
plt.plot(test_loss, 'r-', label = "test_loss")
plt.title("loss")
plt.show()
