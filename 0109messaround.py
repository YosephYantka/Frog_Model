# this file contains the CNN structure for the first trial of a model
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torchvision import transforms
import wandb
import random
from torch import tensor
from torchmetrics.classification import BinaryAccuracy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#initiate wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="frog_model_binary",
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "CNN",
    "dataset": "notata_background",
    "epochs": 10,
    }
)

# The dataloader:
class FrogLoaderDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = image.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# borrowed VGG16 model structure
class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # changed the in_channels from 3 to 1 because my images are greyscale
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4 * 11 * 512, 4096),
            # will need to change the 7 * 7 because the feature maps are not going to be this size
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)#.sigmoid()
        out = self.sigmoid(out)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Hyperparameters and weight initialisation

num_classes = 1
num_epochs = 11 #I changed this to 11 to solve error at 'for epoch' line
batch_size = 32
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005

training_data = FrogLoaderDataset(
    annotations_file='/home/nottom/Documents/LinuxProject/first_model/annotations_file_training.csv',
    img_dir='/home/nottom/Documents/LinuxProject/first_model/img_dir_training_LATEST')
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

valid_data = FrogLoaderDataset(
    annotations_file='/home/nottom/Documents/LinuxProject/first_model/annotations_file_valid.csv',
    img_dir='/home/nottom/Documents/LinuxProject/first_model/img_dir_valid')
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

model = VGG16(num_classes).to(device)
model = model.apply(init_weights)

# criterion = nn.CrossEntropyLoss()  #
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

# Training the model:
total_step = len(train_loader)
offset = random.random() / 5  #wandbstuff
for epoch in tqdm(range(1, num_epochs), desc='epochs', unit='epoch '): #changed range to solve "division by zero error in line below)
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset #wandb stuff
    loss = 2 ** -epoch + random.random() / epoch + offset #wandb stuff
    wandb.log({"acc": acc, "loss": loss})
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device).type(torch.float) / 255
        labels = labels.type(torch.float).to(device)
        labels[labels == 1] = 0 # THESE TWO LINES OF CODE CONVERT THE 1 AND 2 LABELS TO 0 AND 1 FOR THIS BINARY CLASSIFIER
        labels[labels == 2] = 1
        labels = labels[:, None]  # removing this makes it fail

        # Forward pass
        outputs = model(images)
        #torch metrics stuff!
        metric = BinaryAccuracy(threshold=0.5).to(device)
        accuracy = metric(outputs, labels)

        #print((outputs.detach() > 0.5).cpu().numpy().astype(np.uint8).T, labels.cpu().numpy().T)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy = {}'
          .format(epoch, num_epochs, i + 1, total_step, loss.item(), accuracy))

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device).type(torch.float) / 255
            labels = labels.type(torch.float).to(device)
            labels[labels == 1] = 0  # THESE TWO LINES OF CODE CONVERT THE 1 AND 2 LABELS TO 0 AND 1 FOR THIS BINARY CLASSIFIER
            labels[labels == 2] = 1
            labels = labels[:, None]  # removing this makes it fail

            # forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            metric = BinaryAccuracy(threshold=0.5).to(device)
            accuracy = metric(outputs, labels)

            loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        correct = correct/5451
        print('Accuracy of the network on the {} validation images: {}'.format(5451, accuracy))
        print('Loss: {:.4f}'.format(loss.item()))



#when it comes to running the test data through, the test data is in "/home/nottom/Documents/LinuxProject/test_data_new"
#is the 1st, 3rd, 5th, etc. files