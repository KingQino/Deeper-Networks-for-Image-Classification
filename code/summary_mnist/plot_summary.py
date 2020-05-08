# -*- coding: utf-8 -*-
# @Time    : 2020/5/7 1:57 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : plot_summary.py
# @Software: PyCharm
# read files that we will use


# summarize statistics/ design comparisons
# 1. B - B+aug (Goo, Res)
#    result structure
#    a. goo_b, goo_b + aug
#    b. res_b, res_b + aug
# 2. B - M (Goo, Res)
#    result structure
#    a. goo_b, goo_m
#    b. res_b, res_m
# 3. G - R (base)


import json
import numpy as np
import matplotlib.pyplot as plt

# 1>. goo base
# 2>. res base
try:
    cache_file = open('summary3/tr_acc_epoch1.json', 'r')
    cache_contents = cache_file.read()
    goo_tr = json.loads(cache_contents)
    goo_tr = [goo_tr[str(i + 1)] for i in range(30)]

    cache_file = open('summary3/te_acc_epoch1.json', 'r')
    cache_contents = cache_file.read()
    goo_te = json.loads(cache_contents)
    goo_te = [goo_te[str(i + 1)] for i in range(30)]

    cache_file = open('summary3/tr_acc_epoch2.json', 'r')
    cache_contents = cache_file.read()
    res_tr = json.loads(cache_contents)
    res_tr = [res_tr[str(i + 1)] for i in range(30)]

    cache_file = open('summary3/te_acc_epoch2.json', 'r')
    cache_contents = cache_file.read()
    res_te = json.loads(cache_contents)
    res_te = [res_te[str(i + 1)] for i in range(30)]

    cache_file.close()
except:
    print("something bad happens!")

goo = [goo_tr, goo_te]
res = [res_tr, res_te]
legends = ['Goo Train', 'Goo Test', 'Res Train', 'Res Test']


def plot_accuracy_curve(model1, model2, legend, show=True, path='plot.png'):
    num = 30
    x_axis = np.linspace(1, num, num, endpoint=True)
    plt.plot(x_axis, model1[0], color='r', label=legend[0])
    plt.plot(x_axis, model1[1], color='b', label=legend[1])
    plt.plot(x_axis, model2[0], color='darkgreen', linestyle='--', label=legend[2])
    plt.plot(x_axis, model2[1], color='orange', linestyle='--', label=legend[3])
    plt.legend()
    plt.title('Goo vs Res (Base)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


plot_accuracy_curve(goo, res, legends, path='summary3/plot.png')
