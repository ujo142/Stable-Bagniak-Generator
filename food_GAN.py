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
from food_discriminator import FoodDiscriminator
from food_generator import FoodGenerator



ALL_FOOD_IMAGES = []
class FoodGAN(pl.LightningModule):

    def __init__(self, latent_size = 256, learning_rate = 0.0002, bias1 = 0.5, bias2 = 0.999, batch_size = 128):
        super().__init__()
        self.save_hyperparameters()

        # networks
        # data_shape = (channels, width, height)
        self.generator = FoodGenerator()
        self.discriminator = FoodDiscriminator(input_size=64)
        self.normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.validation = torch.randn(self.batch_size, self.latent_size, 1, 1)
   
        # self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, preds, targets):
        return F.binary_cross_entropy(preds, targets)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch
        
        # train generator
        if optimizer_idx == 0:

            # Generate fake images
            fake_random_noise = torch.randn(self.batch_size, self.latent_size, 1, 1)
            fake_random_noise = fake_random_noise.type_as(real_images)
            fake_images = self(fake_random_noise) #self.generator(latent)
            
            # Try to fool the discriminator
            preds = self.discriminator(fake_images)
            targets = torch.ones(self.batch_size, 1)
            targets = targets.type_as(real_images)
            
            loss = self.adversarial_loss(preds, targets)
            self.log('generator_loss', loss, prog_bar=True)

            tqdm_dict = {'g_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output


        # train discriminator
        if optimizer_idx == 1:

            # Pass real images through discriminator
            real_preds = self.discriminator(real_images)
            real_targets = torch.ones(real_images.size(0), 1)
            real_targets = real_targets.type_as(real_images)
            real_loss = self.adversarial_loss(real_preds, real_targets)
            # real_score = torch.mean(real_preds).item()
            
            # Generate fake images
            real_random_noise = torch.randn(self.batch_size, self.latent_size, 1, 1)
            real_random_noise = real_random_noise.type_as(real_images)
            fake_images = self(real_random_noise) #self.generator(latent)

            # Pass fake images through discriminator
            fake_targets = torch.zeros(fake_images.size(0), 1)
            fake_targets = fake_targets.type_as(real_images)
            fake_preds = self.discriminator(fake_images)
            fake_loss = self.adversarial_loss(fake_preds, fake_targets)
            # fake_score = torch.mean(fake_preds).item()
            self.log('discriminator_loss', fake_loss, prog_bar=True)

            # Update discriminator weights
            loss = real_loss + fake_loss
            self.log('total_loss', loss, prog_bar=True)
            #########
            tqdm_dict = {'d_loss': loss}
            output = OrderedDict({
               'loss': loss,
               'progress_bar': tqdm_dict,
               'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate
        bias1 = self.hparams.bias1
        bias2 = self.hparams.bias2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(bias1, bias2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(bias1, bias2))
    
        return [opt_g, opt_d], []

    def denormalize(self, input_image_tensors):
        input_image_tensors = input_image_tensors * self.normalize[1][0]
        input_image_tensors = input_image_tensors + self.normalize[0][0]
        return input_image_tensors

    def save_generated_samples(self, index, sample_images):
        fake_fname = 'generated-images-{}.png'.format(index)
        save_image(self.denormalize(sample_images[-64:]), os.path.join(".", fake_fname), nrow=8)

    def on_epoch_end(self):
        # import pdb;pdb.set_trace()
        z = self.validation.type_as(self.generator.model[0].weight)
        sample_imgs = self(z) #self.current_epoch
        #ALL_FOOD_IMAGES.append(sample_imgs.cpu())
        self.save_generated_samples(self.current_epoch, sample_imgs)