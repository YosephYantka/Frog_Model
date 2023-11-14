import os
import csv
import glob
from pathlib import Path
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import re
import math
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import shutil
import tqdm
import PIL
from random import sample

#assessing how many multi-label examples I have
import shutil

x=0
folder = '/home/nottom/Documents/Training/training_data/BACKUP/multi_labelled_examples/5' # make sure to do both training and validation text directories
for file in glob.glob('/home/nottom/Documents/Training/training_data/BACKUP/multi_labelled_examples/5/**/*.txt', recursive=True):
    join_path = os.path.join(folder, file)
    reader = open(join_path, 'r')
    content = reader.read()
    # original = '/home/nottom/Documents/Training/training_data/6/text/' + file
    # destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/6/' + file
    # original_png = '/home/nottom/Documents/Training/training_data/6/specgrams/' + file[:-4]
    # destination_png = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/6/' + file[-4]
    if content == "0, 0, 0, 0, 0, 1, 1":
        x += 1
print(x)

        # shutil.move(original, destination)
        # shutil.move(original_png, destination_png)


#removing the .png in new text files
folder = '/home/nottom/Documents/LinuxProject/multi_class_model/text_directories/text_dir_training'
#this one to move files
for file in glob.glob('/home/nottom/Documents/LinuxProject/multi_class_model/text_directories/text_dir_training/**/*.txt', recursive=True):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    # print(file)
    # original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    # destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H1_1506_20190817_205359/' + file
    # if file.endswith('1506__0__20190817_205359_0_.txt'):
    #     shutil.move(original, destination)
    # print(file[-6:])
    if file[-7:] == 'png.txt':
        newname = file[:-8] + '.txt'
        print(newname)
        os.rename(file, newname)

x = 0
for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/**/*.wav', recursive=True):
    x += 1

print(x)