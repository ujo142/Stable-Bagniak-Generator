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






class FoodGenerator(nn.Module):
    def __init__(self, latent_size = 256):
        super().__init__()
        self.latent_size = latent_size
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.bias = False

        self.model = nn.Sequential(

              #input size: (latent_size,1,1)
              nn.ConvTranspose2d(latent_size, 512, kernel_size=self.kernel_size, stride=1, padding=0, bias=self.bias),
              nn.BatchNorm2d(512),
              nn.ReLU(True),

              #input size: (512,4,4)
              nn.ConvTranspose2d(512, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias),
              nn.BatchNorm2d(256),
              nn.ReLU(True),

              #input size: (256,8,8)
              nn.ConvTranspose2d(256, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias),
              nn.BatchNorm2d(128),
              nn.ReLU(True),

              #input size: (128,16,16)
              nn.ConvTranspose2d(128, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias),
              nn.BatchNorm2d(64),
              nn.ReLU(True),

              nn.ConvTranspose2d(64, 3, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=self.bias),
              nn.Tanh()
              # output size: 3 x 64 x 64
        )

    def forward(self, input_img):
        input_img = self.model(input_img)
        return input_img