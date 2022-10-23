import opendatasets as od
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import torch
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.utils import save_image
import os






class FoodDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size
        self.channel = 3
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.bias = False
        self.negative_slope = 0.2

        #input size: (3,64,64)
        self.conv1 = nn.Conv2d(self.channel, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.LeakyReLU(self.negative_slope, inplace=True)

        #input size: (128,32,32)
        self.conv2 =  nn.Conv2d(128, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(256)

        #input size: (256,16,16)
        self.conv3 =  nn.Conv2d(256, 512, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(512)

        #input size: (512,8,8)
        self.conv4 =  nn.Conv2d(512, 1024, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias)
        self.bn4 = nn.BatchNorm2d(1024)

        self.fc = nn.Sequential(
            nn.Linear(in_features=16384,out_features=1),
            nn.Sigmoid()
        )


    def forward(self, input_img):
        validity = self.conv1(input_img)
        validity = self.bn1(validity)
        validity = self.relu(validity)
        validity = self.conv2(validity)
        validity = self.bn2(validity)
        validity = self.relu(validity)
        validity = self.conv3(validity)
        validity = self.bn3(validity)
        validity = self.relu(validity)
        validity = self.conv4(validity)
        validity = self.bn4(validity)
        validity = self.relu(validity)
        validity=validity.view(-1, 1024*4*4)
        validity=self.fc(validity)
        return validity