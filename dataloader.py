import os
import pandas as pd
from PIL import ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.io
from torchvision.io import read_image
from pathlib import Path
from typing import Tuple
import torch
import numpy as np

# import cv2


# Dataloader class:
class FrogLoaderDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]


        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label


# Loading the data and creating the data loader:
training_data = FrogLoaderDataSet(
    annotations_file='/home/nottom/Documents/LinuxProject/first_model/annotations_file_training.csv',
    img_dir='/home/nottom/Documents/LinuxProject/first_model/img_dir_training', )

training_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

# (optional) Iterate through the data loader

# IT IS IMPORTANT IN THIS STEP TO DELETE ALL SEGMENTS (DON'T FORGET TEXT FILES!!) IN THE DIRECTORIES THAT ARE BEGIN WITH 3591 OR 3594 (using pillow (Image) to
# find sizes and print file names of abnormally sized segments)
# ^this code is in spare.py

for i, sample in enumerate(training_dataloader):
    print(i, sample[0].shape, sample[1])
    print(type(sample[0]))
    # plt.imshow(sample[0])
    # plt.waitforbuttonpress()

img = '/home/nottom/Documents/LinuxProject/first_model/img_dir_training/0_4_1193__0__20150617_231301_0_.png'

print(img.shape)


# Stefan's version of the dataloader:
class FrogLoaderDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        img_path = self.img_dir / self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        # img_path = self.img_dir / self.img_labels.iloc[idx, 0]
        # image = plt.imread(str(img_path))[:, :, :3]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

#
# img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
