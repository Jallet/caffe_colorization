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

train_loss = np.loadtxt(train_loss_path)
test_loss = np.loadtxt(test_loss_path)
plt.figure(1)
plt.plot(train_loss[:, 0], train_loss[:, 1], 'g-', label = "train_loss")
plt.plot(test_loss[:, 0], test_loss[:, 1], 'r-', label = "test_loss")
plt.legend()
plt.title("loss")
plt.show()
