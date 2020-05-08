# -*- coding: utf-8 -*-
# @Time    : 2020/5/7 3:03 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : plot_B_Aug.py
# @Software: PyCharm

# summarize statistics/ design comparisons
# 1. B - B+aug (Goo, Res)
#    result structure
#    a. goo_b, goo_b + aug
#    b. res_b, res_b + aug
# 2. B - M (Goo, Res)
#    result structure
#    . res_b, res_m
# 3. G - R (base)

import json
import numpy as np
import matplotlib.pyplot as plt

# 1>. goo/res base
# 2>. goo/res base+aug
try:
    cache_file = open('summary1a/tr_acc_epoch1.json', 'r')
    cache_contents = cache_file.read()
    goo_b_tr = json.loads(cache_contents)
    goo_b_tr = [goo_b_tr[str(i + 1)] for i in range(30)]

    cache_file = open('summary1a/te_acc_epoch1.json', 'r')
    cache_contents = cache_file.read()
    goo_b_te = json.loads(cache_contents)
    goo_b_te = [goo_b_te[str(i + 1)] for i in range(30)]

    cache_file = open('summary1a/tr_acc_epoch2.json', 'r')
    cache_contents = cache_file.read()
    goo_ba_tr = json.loads(cache_contents)
    goo_ba_tr = [goo_ba_tr[str(i + 1)] for i in range(30)]

    cache_file = open('summary1a/te_acc_epoch2.json', 'r')
    cache_contents = cache_file.read()
    goo_ba_te = json.loads(cache_contents)
    goo_ba_te = [goo_ba_te[str(i + 1)] for i in range(30)]

    cache_file.close()
except:
    print("something bad happens!")

base =     [goo_b_tr, goo_b_te]
base_aug = [goo_ba_tr, goo_ba_te]
legends = ['Goo b Train', 'Goo b Test', 'Goo b+a Train', 'Goo b+a Test']
# legends = ['Res b Train', 'Res b Test', 'Res b+a Train', 'Res b+a Test']


def plot_accuracy_curve(model1, model2, legend, show=True, path='plot.png'):
    num = 30
    x_axis = np.linspace(1, num, num, endpoint=True)
    plt.plot(x_axis, model1[0], color='r', label=legend[0])
    plt.plot(x_axis, model1[1], color='b', label=legend[1])
    plt.plot(x_axis, model2[0], color='darkgreen', linestyle='--', label=legend[2])
    plt.plot(x_axis, model2[1], color='orange', linestyle='--', label=legend[3])
    plt.legend()
    plt.title('Base vs Base+Aug (GoogLeNet) CIFAR-10')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


plot_accuracy_curve(base, base_aug, legends, path='summary1a/plot.png')
