# -*- coding: utf-8 -*-
# @Time    : 2020/4/29 10:33 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : utils.py
# @Software: PyCharm
# Reference:
#   https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_learning
#   https://github.com/39239580/googlenet-pytorch/blob/master/Inception_v1_mnist.py
#   https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_curve(train_list, test_list, show=False, save=True, path='results/accuracy_epoch.png'):
    """
    It is used to plot the accuracy curve. Note that train_list and test_list have the same size.

    :param train_list: the accuracy list containing accuracies per epoch for train data
    :param test_list: the accuracy list containing accuracies per epoch for test data
    :param show: whether to show the plot
    :param save: whether to save the plot
    :param path: if the plot is saved, the path that will be saved
    :return: no
    """
    num = len(train_list)
    x_axis = np.linspace(1, num, num, endpoint=True)
    plt.plot(x_axis, train_list, color='r', label='Train accuracy')
    plt.plot(x_axis, test_list, color='b', label='Test accuracy')
    plt.legend()
    plt.title('Accuracy against epoch in the train and test sets')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    if save:
        if not os.path.exists('results'):
            os.mkdir('results')
        else:
            pass
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    plt.close()


def plot_loss_curve(train_list, test_list, show=False, save=True, path='results/loss_epoch.png'):
    """
    It is used to plot the accuracy curve. Note that train_list and test_list have the same size.

    :param train_list: the loss list containing losses per epoch for train data
    :param test_list: the loss list containing losses per epoch for train data
    :param show: whether to show the plot
    :param save: whether to save the plot
    :param path: if the plot is saved, the path that will be saved
    :return: no
    """
    num = len(train_list)
    x_axis = np.linspace(1, num, num, endpoint=True)
    plt.plot(x_axis, train_list, color='r', label='Train loss')
    plt.plot(x_axis, test_list, color='b', label='Test loss')
    plt.legend()
    plt.title('Loss against epoch in the train and test sets')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if save:
        if not os.path.exists('results'):
            os.mkdir('results')
        else:
            pass
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    plt.close()


def get_confusion_matrix(classes, predicts_list, labels_list):
    """
    Given predicted results and ground-truth labels, the function return the corresponding confusion matrix.

    :param classes: the classes of labels, it is used to compute the length of the classes
    :param predicts_list: the list of predicted results
    :param labels_list: the list of label results
    :return: confusion matrix
    """
    size = len(classes)
    length = len(predicts_list)
    confusion_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            confusion_matrix[i][j] = np.sum([(predicts_list[k] == i) and (labels_list[k] == j) for k in range(length)])
    return confusion_matrix


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    """
    Given a sklearn confusion matrix (cm), make a nice plot

    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: given classification classes such as [0, 1, 2], the class names,
                         for example: ['high', 'medium', 'low']
    :param title: the text to display at the top of the matrix
    :param cmap: the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    :param normalize: If False, plot the raw numbers
                  If True, plot the proportions
    :return: no
    Usage
    -----
        plot_confusion_matrix(cm  = cm,                  # confusion matrix created by sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    # the result will be stored at
    if not os.path.exists('results'):
        os.mkdir('results')
    else:
        pass
    plt.savefig('results/confusion_matrix.png')
    plt.show()

