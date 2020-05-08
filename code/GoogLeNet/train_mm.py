# -*- coding: utf-8 -*-
# @Time    : 2020/5/3 12:09 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : train_mm.py
# @Software: PyCharm
# Reference:
#   https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_learning
#   https://github.com/39239580/googlenet-pytorch/blob/master/Inception_v1_mnist.py
#   https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import numpy as np
import os
from torchvision import transforms, datasets
from torch.autograd import Variable
from modified_model import GoogLeNet_MNIST
from utils import get_confusion_matrix, plot_confusion_matrix, plot_loss_curve, plot_accuracy_curve

# select the device we are going to use. GPU(cuda) preferred
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# parameters
learning_rate = 0.01
batch_size = 64
num_epoch = 30  # the number of training epoch
num_classes = 10
momentum = 0.9

# Get MNIST dataset and preprocess it
data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),  # [0, 255] --> [0, 1]
                                 transforms.Normalize((0.5,), (0.5,))]),  #  [0, 1] -->[-1, 1]
    "test": transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])}

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=data_transform["train"])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=data_transform["test"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Initial model. Set the loss function, optimizer, learning rate. Applying auxiliary classifiers was thought to
# combat the vanishing gradient problem while providing regularization.
# net = GoogLeNet(num_classes=num_classes, aux_logits=True, init_weights=True)
net = GoogLeNet_MNIST(num_classes=num_classes, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

# Initial the statistical values.
train_accuracy = []  # the accuracy list contains the predicted accuracy of the model (obtained per epoch) in train data
test_accuracy = []  # the accuracy list contains the predicted accuracy of the model (obtained per epoch) in test data
train_loss = []  # the loss list contains the loss of the model (obtained per epoch) in train data
test_loss = []  # the loss list contains the loss of the model (obtained per epoch) in test data
time_per_epoch = []  # the time spent per epoch
best_accuracy = 0.0  # the model achieving the best accuracy in test data will be stored in epoch iteration
save_path = './googleNet_modified.pth'  # the path the model will save
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    # train
    net.train()
    for step, (images, labels) in enumerate(train_loader, start=0):
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # forward
        logits = net(images)
        loss = loss_function(logits, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r[Epoch {}]train process: {:^3.0f}%[{}->{}] train loss: {:.3f}".format(epoch + 1, int(rate * 100), a,
                                                                                       b, loss.item()), end="")
    print()
    epoch_training_time = time.time()

    # test
    net.eval()
    # Compute the accuracy and loss in the train data.
    # Eval model only have last output layer. Note that len(train_dataset) and len(train_loader) are different, the
    # first one is the number of images in train dataset and the second one is the number of batches.
    corr_num = 0  # accumulate the number of correct predicted outputs
    running_loss = 0.0  # accumulate the loss for all over the data
    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            outputs = net(images)
            loss_ = loss_function(outputs, labels)
            running_loss += loss_.item()
            predict_y = torch.max(outputs, dim=1)[1]
            corr_num += (predict_y == labels).sum().item()
        accuracy = corr_num / len(train_dataset)
        loss_ = running_loss / len(train_loader)
        train_accuracy.append(100.0 * accuracy)
        train_loss.append(loss_)
        print('  ｜train loss: %.3f  ｜ train accuracy: %.3f ｜' % (loss_, accuracy))

    # Compute the accuracy and loss in the test data.
    corr_num = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            outputs = net(images)
            loss_ = loss_function(outputs, labels)
            running_loss += loss_.item()
            predict_y = torch.max(outputs, dim=1)[1]
            corr_num += (predict_y == labels).sum().item()
        accuracy = corr_num / len(test_dataset)
        loss_ = running_loss / len(test_loader)
        test_accuracy.append(100.0 * accuracy)
        test_loss.append(loss_)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(net.state_dict(), save_path)
        print('  ｜test  loss: %.3f  ｜ test  accuracy: %.3f ｜' % (loss_, accuracy))

    epoch_end_time = time.time()
    whole_time = epoch_end_time - epoch_start_time
    train_time = epoch_training_time - epoch_start_time
    time_per_epoch.append(whole_time)
print('Finished Training! The total time cost is: %.2fmin | The average time per epoch is: %.3fmin' %
      (np.sum(time_per_epoch) / 60, np.sum(time_per_epoch) / num_epoch / 60))
print('The best accuracy is %.3f' % best_accuracy)

tr_acc_dict = dict((key, val) for key, val in enumerate(train_accuracy, start=1))
te_acc_dict = dict((key, val) for key, val in enumerate(test_accuracy, start=1))
tr_los_dict = dict((key, val) for key, val in enumerate(train_loss, start=1))
te_los_dict = dict((key, val) for key, val in enumerate(test_loss, start=1))
# write dict into json file
json_str1 = json.dumps(tr_acc_dict, indent=4)
json_str2 = json.dumps(te_acc_dict, indent=4)
json_str3 = json.dumps(tr_los_dict, indent=4)
json_str4 = json.dumps(te_los_dict, indent=4)
if not os.path.exists('results'):
    os.mkdir('results')
with open('results/tr_acc_epoch.json', 'w') as json_file:
    json_file.write(json_str1)
with open('results/te_acc_epoch.json', 'w') as json_file:
    json_file.write(json_str2)
with open('results/tr_los_epoch.json', 'w') as json_file:
    json_file.write(json_str3)
with open('results/te_los_epoch.json', 'w') as json_file:
    json_file.write(json_str4)

# plot the accuracy and loss figures
plot_accuracy_curve(train_accuracy, test_accuracy, show=True)
plot_loss_curve(train_loss, test_loss, show=True)
# plot confusion matrix of the predicted results of the latest model
net.eval()
predict_list = np.array([]).astype(int)
label_list = np.array([]).astype(int)
with torch.no_grad():
    for images, labels in test_loader:
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))
        outputs = net(images)
        predict_y = torch.max(outputs, dim=1)[1]
        predict_list = np.hstack((predict_list, predict_y.cpu().numpy()))
        label_list = np.hstack((label_list, labels.cpu().numpy()))
cm = get_confusion_matrix(classes, predict_list, label_list)
plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=True)
