# this file contains the CNN structure for the first trial of a model
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import pandas as pd
import os
import matplotlib as mpl
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
from torchvision import transforms
import wandb
import random
from torch import tensor
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

# Device configuration
mpl.use("TkAgg")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(mpl.get_backend())
# os.chdir('/home/nottom/Documents/LinuxProject/first_model')

num_classes = 7
num_epochs = 10  # I changed this to 11 to solve error at 'for epoch' line
batch_size = 32
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005

# #initiate wandb
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="frog_model_binary",
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.001,
#     "architecture": "CNN",
#     "dataset": "notata_background",
#     "epochs": 10,
#     }
# )

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
        label_0 = self.img_labels.iloc[idx, 1]
        label_1 = self.img_labels.iloc[idx, 2]
        label_2 = self.img_labels.iloc[idx, 3]
        label_3 = self.img_labels.iloc[idx, 4]
        label_4 = self.img_labels.iloc[idx, 5]
        label_5 = self.img_labels.iloc[idx, 6]
        label_6 = self.img_labels.iloc[idx, 7]
        filename = self.img_labels.iloc[idx, 0]
        image = image.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_0 = self.target_transform(label_0)
            label_1 = self.target_transform(label_1)
            label_2 = self.target_transform(label_2)
            label_3 = self.target_transform(label_3)
            label_4 = self.target_transform(label_4)
            label_5 = self.target_transform(label_5)
            label_6 = self.target_transform(label_6)
        return image, label_0, label_1, label_2, label_3, label_4, label_5, label_6, filename


# borrowed VGG16 model structure
class VGG16(nn.Module):
    def __init__(self, num_classes=num_classes):
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
        out = self.fc2(out)  # .sigmoid()
        out = self.sigmoid(out)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# Hyperparameters and weight initialisation

num_classes = 7
num_epochs = 10  # I changed this to 11 to solve error at 'for epoch' line
batch_size = 32
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005

# training_data = FrogLoaderDataset(
#     annotations_file='/home/nottom/Documents/LinuxProject/first_model/annotations_file_training.csv',
#     img_dir='/home/nottom/Documents/LinuxProject/first_model/img_dir_training_LATEST')
# train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
#
# valid_data = FrogLoaderDataset(
#     annotations_file='/home/nottom/Documents/LinuxProject/first_model/annotations_file_valid.csv',
#     img_dir='/home/nottom/Documents/LinuxProject/first_model/img_dir_valid')
# valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

test_data = FrogLoaderDataset(
    annotations_file='/home/nottom/Documents/LinuxProject/multi_class_model/annotations_file_test_multi.csv',
    img_dir='/home/nottom/Documents/LinuxProject/multi_class_model/img_dir_test',)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True) #remember to change drop_last if I need to!!

model = VGG16(num_classes).to(device)
model = model.apply(init_weights)

# criterion = nn.CrossEntropyLoss()  #
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

from torch import rand, randint
from torchmetrics.classification import MultilabelPrecisionRecallCurve
# evaluate model on test dataset:
# numbers = (4, 5, 6, 7, 8, 9, 10)
def test(model, device, test_loader):
    # threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for threshold in threshold_list:
        model.load_state_dict(
            torch.load("/home/nottom/Documents/LinuxProject/multi_class_model/multiclass_model.pt"))
        # # model.load_state_dict( torch.load("/home/nottom/Documents/LinuxProject/first_model/second_model_version_epoch_4.pt")
        model.eval()
        test_loss = 0
        correct = 0
        running_accuracy = 0
        running_precision = 0
        running_recall = 0
        threshold = 0.9
        long_outputs = torch.empty(0,7).to(device)
        long_labels = torch.empty(0,7).to(device)
        with torch.no_grad():
            for i, (images, labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6, filenames) in enumerate(test_loader):
                # Move tensors to the configured device
                images = images.to(device).type(torch.float) / 255
                labels_0 = labels_0.type(torch.float).to(device)
                labels_1 = labels_1.type(torch.float).to(device)
                labels_2 = labels_2.type(torch.float).to(device)
                labels_3 = labels_3.type(torch.float).to(device)
                labels_4 = labels_4.type(torch.float).to(device)
                labels_5 = labels_5.type(torch.float).to(device)
                labels_6 = labels_6.type(torch.float).to(device)
                labels_0 = labels_0[:, None]
                labels_1 = labels_1[:, None]
                labels_2 = labels_2[:, None]
                labels_3 = labels_3[:, None]
                labels_4 = labels_4[:, None]
                labels_5 = labels_5[:, None]
                labels_6 = labels_6[:, None]  # removing this makes it fail

                # forward pass
                labels = torch.cat((labels_0, labels_1, labels_2, labels_3, labels_4, labels_5, labels_6), -1)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                rounded_outputs = torch.round(outputs)

                #precision-recall curve
                long_outputs = torch.cat((long_outputs, outputs), 0).to(device)
                long_labels = torch.cat((long_labels, labels), 0).to(device)

                # torch metrics
                metric1 = BinaryAccuracy(threshold=threshold).to(device)
                accuracy = metric1(outputs, labels)
                metric2 = BinaryPrecision(threshold=threshold).to(device)
                precision = metric2(outputs, labels)
                metric3 = BinaryRecall(threshold=threshold).to(device)
                recall = metric3(outputs, labels)
                loss = criterion(outputs, labels)
                running_accuracy += accuracy
                running_precision += precision
                running_recall += recall

                # # # output filenames, predicted, labels
                # for data in range(batch_size):
                #     with open('/home/nottom/Documents/LinuxProject/multi_class_model/output/' + str(filenames[data]) + '.txt',
                #           'x') as f:
                #         f.write(str(filenames[data]))
                #         stroutputs = str(rounded_outputs[data])
                #         f.write("," + stroutputs[8] + ',' + stroutputs[12] + ',' + stroutputs[16] + ',' + stroutputs[20]
                #                 + ',' + stroutputs[24] + ',' + stroutputs[28] + ',' + stroutputs[32])
                #         strlabels = str(labels[data])
                #         f.write("," + strlabels[8]+ "," + strlabels[12] + "," + strlabels[16] + "," + strlabels[20] +
                #                 "," + strlabels[24] + "," + strlabels[28] + "," + strlabels[32])

            print('threshold {}, - TEST SET: Accuracy: {}, Loss: {:.4f}, Precision: {}, Recall: {}'.format(
                                                                                                   threshold,
                                                                                                   running_accuracy / 30,
                                                                                                   loss.item(),
                                                                                                   running_precision / 30,
                                                                                                   running_recall / 30))
            long_labels = long_labels.type(torch.LongTensor).to(device)
            # long_outputs = long_outputs.type(torch.LongTensor).to(device)
            metric = MultilabelPrecisionRecallCurve(num_labels=7, thresholds=None).to(device)
            # precision, recall, thresholds = mlprc(long_outputs, long_labels)
            metric.update(long_outputs, long_labels)
            mpl.pyplot.show()
            fig_, ax_ = metric.plot(score=True)


test(model, 'cuda', test_loader)



#precision-recall curve:





# # successfully constructed a csv output!!
# import os
# import csv
# from pathlib import Path
# import pandas as pd
# folder = '/home/nottom/Documents/LinuxProject/multi_class_model/output'
# os.chdir('/home/nottom/Documents/LinuxProject/multi_class_model/output')
# with open('multiclass_model_predictions.csv', 'w') as out_file:
#     csv_out = csv.writer(out_file)
#     column_names = ['filename', 'prediction_0', 'label_0',
#                                 'prediction_1', 'label_1',
#                                 'prediction_2', 'label_2',
#                                 'prediction_3', 'label_3',
#                                 'prediction_4', 'label_4',
#                                 'prediction_5', 'label_5',
#                                 'prediction_6',' label_6']
#     writer = csv.DictWriter(out_file, fieldnames=column_names)
#     writer.writeheader()
#     # csv_out.writerow(['FileName', 'Content'])
#     for fileName in Path('.').glob('*.txt'):
#         # lala = fileName
#         # csv_out.writerow([str(fileName) + ',png',open(str(fileName.absolute())).read().strip()])
#         name = open(str(fileName.absolute())).read()
#         # name = name.replace('[', '')
#         # name = name.replace(']', '')
#         # name = name[:-2]
#         name_split = name.split(',')
#         # print(name_split)
#         # threshold = "," + str(round(float(name_split[1]))) #edit if outputting csv and not using 0.5 as threshold!!!
#         # name += threshold
#         # print(name_split[0][15:23])
#         # if (name_split[0][15:23]) == "20210723":
#         csv_out.writerow([name_split[0], name_split[1], name_split[8], name_split[2], name_split[9], name_split[3]
#                           , name_split[10], name_split[4], name_split[11], name_split[5], name_split[12], name_split[6]
#                           , name_split[13], name_split[7], name_split[14]])
#             # print(name_split[0])
#
