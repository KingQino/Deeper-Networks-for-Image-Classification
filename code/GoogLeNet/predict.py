# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 2:30 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : predict.py
# @Software: PyCharm
# Reference:
#   https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_learning
#   https://github.com/39239580/googlenet-pytorch/blob/master/Inception_v1_mnist.py
#   https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
import torch
from model import GoogLeNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("../tulip.jpg")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = GoogLeNet(num_classes=10, aux_logits=False)
# load model weights
model_weight_path = "./googleNet.pth"
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
model.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_indict[str(predict_cla)])
plt.show()


