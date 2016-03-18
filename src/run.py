#!/usr/bin/env python
import subprocess
import re
import numpy as np
project_root = '/home/jiangliang/code/caffe_colorization/'
nohup_file = 'nohup.out'
train_loss_string = '\"Train net output\"'
val_loss_string = '\"Test net output\"'
train_iter_str = "\"Iteration.*loss.*\""
val_iter_str = "\"Iteration.*Testing.*\""
child = subprocess.Popen(project_root + 'src/solve.py', shell = True)
child.wait()

def get_loss(loss_string, iter_string):
    loss_output = subprocess.check_output('cat ' + project_root + nohup_file + ' | grep ' + loss_string, shell = True)
    iter_output = subprocess.check_output('cat ' + project_root + nohup_file + ' | grep ' + iter_string, shell = True)
    
    print loss_output
    tmp_loss = loss_output.split('\n')
    tmp_iter = iter_output.split('\n')
    loss = np.zeros((len(tmp_loss) - 1, 2))
    for i in range(len(tmp_loss) - 1):
        s = re.search('[0-9] = .*loss', tmp_loss[i])
        t = s.group(0)
        s = re.search('Iteration.*,', tmp_iter[i])
        p = s.group(0)
        loss[i, :] = np.array([int(p[10 : -1]), float(t[4 : -5])])

    return loss

train_loss = get_loss(train_loss_string, train_iter_str)
val_loss = get_loss(val_loss_string, val_iter_str)

np.savetxt(project_root + 'result/train_loss', train_loss)
np.savetxt(project_root + 'result/val_loss', val_loss)

child = subprocess.Popen('rm -rf ' + project_root + nohup_file, shell = True)
child.wait()
