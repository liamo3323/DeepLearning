
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10
import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image



def get_train_images(num):
    return torch.stack([train_dataset[i][0] for i in range(num)], dim=0)

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x
    
class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
        # You code goes here.
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2*16*c_hid),
            act_fn(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
    
class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 decoder_class : object = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title(f"Reconstructed from model")
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def train_one_epoch():
    running_loss = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        loss = model._get_reconstruction_loss(inputs)
        loss.backward()
        running_loss += loss
        optimizer.step()
        print(f"Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}", end="\r", flush=True)
    return running_loss/len(train_loader)

if __name__ == "__main__": 

    DATASET_PATH = "dataset"

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", torch.cuda.get_device_name(device))

    # Transformations applied on each image => only make them a tensor
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

    # for i in range(2):
    #     # Load example image
    #     img, _ = train_dataset[i]
    #     img_mean = img.mean(dim=[1,2], keepdims=True)

    #     # Shift image by one pixel
    #     SHIFT = 1
    #     img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
    #     img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
    #     img_shifted[:,:1,:] = img_mean
    #     img_shifted[:,:,:1] = img_mean
    #     compare_imgs(img, img_shifted, "Shifted -")

    #     # Set half of the image to zero
    #     img_masked = img.clone()
    #     img_masked[:,:img_masked.shape[1]//2,:] = img_mean
    #     compare_imgs(img, img_masked, "Masked -")

    epoch_number = 0
    # latest_dim = [1024, 2048, 4096]
    # list_base_channel_size = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    # best_vloss = 1_000_000.
    # for base_channel in list_base_channel_size: 
    #     for dim in latest_dim:
    model = Autoencoder(base_channel_size=256, latent_dim=4096)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # your code here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs') 
    model.train()

    EPOCHS = 50

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()
        running_vloss = 0.0
        model.eval()
        # print(f"{avg_loss}")
    #     epoch_number += 1
    # if avg_loss < best_vloss:
    #     best_vloss = avg_loss
    #     torch.save(model.state_dict(), f"model_{epoch}_{dim}_{base_channel}.pt")
    #     print(f"Model saved at epoch {epoch_number} with loss: {best_vloss:.4f} latent_dim: {dim} base_channel_size: {base_channel}")


    input_imgs = get_train_images(4)
    visualize_reconstructions(model, input_imgs)