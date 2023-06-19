# this file chunks specified wav files, and creates and exports text documents and spectrograms for each chunk
import torch
from torch.utils.data import Dataset
from torchaudio import datasets
import torchaudio.functional as F
import torchaudio.transforms as T
import re
import pandas as pd
from pydub import AudioSegment
import math
import matplotlib
import librosa
import matplotlib.pyplot as plt
import os
import shutil
import wandb
import random

# # start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
#
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     }
# )
# # simulate training
# epochs = 10
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
#     loss = 2 ** -epoch + random.random() / epoch + offset
#
#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})
#
# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()
#^WEIGHTS AND BIASES STUFF^^

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

