#import opendatasets as od
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

from food_discriminator import FoodDiscriminator
from food_generator import FoodGenerator
from food_GAN import FoodGAN

def main():
    # Dataset download
    #dataset_url = 'https://www.kaggle.com/trolukovich/food11-image-dataset'
    #od.download(dataset_url)


    # Images 
    image_size = 64
    batch_size = 256
    normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    latent_size = 256
    food_data_directory = '/Users/ben/python_projects/bagniak_gen/Stable-Bagniak-Generator/food11-image-dataset/training'

    # Pack data into dataloader
    food_train_dataset = ImageFolder(food_data_directory, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*normalize)]))
    food_train_dataloader = DataLoader(food_train_dataset, batch_size, num_workers=0, pin_memory=True, shuffle=True)
    

    model = FoodGAN()
    trainer = pl.Trainer( max_epochs=10, accelerator="cpu")
    trainer.fit(model, food_train_dataloader)

if __name__ == "__main__":
    main()