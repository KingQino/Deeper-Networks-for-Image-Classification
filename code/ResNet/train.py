# -*- coding: utf-8 -*-
# @Time    : 2020/5/5 10:24 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : train.py
# @Software: PyCharm
# Reference:
#   https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_learning
#   https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import time
import torch.optim as optim
from modified_model import resnet34
from utils import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

# parameters
learning_rate = 0.0001
batch_size = 32
num_epoch = 30  # the number of training epoch
num_classes = 10


# Get MNIST dataset and preprocess it
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # interpolation=2   PIL.Image.BILINEAR
                                 transforms.ToTensor(),  # [0, 255] --> [0, 1]
                                 transforms.Normalize((0.5,), (0.5,))]),  # [0, 1] -->[-1, 1]
    "test": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])}

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=data_transform["train"])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=data_transform["test"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# # Get cifar10 dataset and preprocess it
# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#     "test": transforms.Compose([transforms.Resize(256),
#                                transforms.CenterCrop(224),
#                                transforms.ToTensor(),
#                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#
# train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transform["train"])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transform["test"])
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# store the names of classes
data_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in data_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('data/class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# Initial model.
net = resnet34(num_classes=num_classes)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Initial the statistical values.
train_accuracy = []  # the accuracy list contains the predicted accuracy of the model (obtained per epoch) in train data
test_accuracy = []  # the accuracy list contains the predicted accuracy of the model (obtained per epoch) in test data
train_loss = []  # the loss list contains the loss of the model (obtained per epoch) in train data
test_loss = []  # the loss list contains the loss of the model (obtained per epoch) in test data
time_per_epoch = []  # the time spent per epoch
best_accuracy = 0.0
save_path = './resNet34.pth'
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
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

    # validate
    net.eval()
    # Compute the accuracy and loss in the train data.
    corr_num = 0  # accumulate the number of correct predicted outputs
    running_loss = 0.0  # accumulate the loss for all over the data
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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


# store the accuracy and loss statistics
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
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        predict_y = torch.max(outputs, dim=1)[1]
        predict_list = np.hstack((predict_list, predict_y.cpu().numpy()))
        label_list = np.hstack((label_list, labels.cpu().numpy()))
cm = get_confusion_matrix(classes, predict_list, label_list)
plot_confusion_matrix(cm, classes, title='Confusion matrix', normalize=True)
