# this file chunks specified wav files, and creates and exports text documents and spectrograms for each chunk
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import re
import math
import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import shutil
import PIL
from random import sample
import os

#for finding incorrectly labelled text files in label directory :
folder = '/home/nottom/Documents/LinuxProject/training_data_2009/all_textfiles'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    if content != '0' and content != '1': # and content != '2' and content != '3' and content != '4' and content != '5' and content != '6'
        print(file)
        print(content)

# use this to check if all images are the same size (THEY AREN'T)
folder = '/home/nottom/Documents/LinuxProject/training_data_2009/all_spectrograms'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    img = PIL.Image.open(join_path)
    width = img.width
    height = img.height
    # print(height, width)
    if width != 376:
        print(file)
    # if height != 128:
    #     print(file)


#For creating balanced datasets - culls total background directory and creates a corresponding text directory
#culling
import os
from random import sample
import random
import shutil

folder = '/home/nottom/Documents/LinuxProject/first_model/test_data/background_spectrograms' #need this line and below line!!
files = os.listdir('/home/nottom/Documents/LinuxProject/first_model/test_data/background_spectrograms') #need this!!
for file in sample(files,4645):
    path = os.path.join(folder, file)
    os.unlink(path)

#creates new corresponding text file
import os
import PIL
import shutil
folder = '/home/nottom/Documents/LinuxProject/first_model/test_data/background_spectrograms'
for file in os.listdir(folder):
    if file.endswith('_0_.png'):
        with open('/home/nottom/Documents/LinuxProject/first_model/test_data/test_text/' + file[:-4] + '.txt', 'x') as f:
            f.write("1")
    if file.endswith('_1_.png'):
        with open('/home/nottom/Documents/LinuxProject/first_model/test_data/test_text/' + file[:-4] + '.txt', 'x') as f:
            f.write("2") #1 and 2 are necessary for this code to work with current VGG16 structure


#file deleter - CURSED
folder = '/home/nottom/Documents/LinuxProject/first_model/training_data_text'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if filename.endswith('.png'):
        os.unlink(file_path)

#THIS ONE WORKEDDDD - for sorting files in one folder into a number of smaller files based on filename
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/miscandfashionmnist/whatever/' + file
    destination = '/home/nottom/Documents/LinuxProject/miscandfashionmnist/h4/' + file
    if file.endswith('1193__0__20170526_202547_0_.txt'):
        shutil.move(original, destination)


 folder = '/home/nottom/Documents/LinuxProject/test_data/spectrograms/backup_background'

#this one to move files
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/spectrograms/backup_background/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/spectrograms/trimmed/H2BAR2_20210710_221000/' + file
    if file.endswith('BAR2_20210710_221000_0_.png'):
        shutil.move(original, destination)

    if content != '1' and '2':
        print(file)






















folder = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup'
#this one to move files
for file in range(1, 218(os.listdir(folder))):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    # original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    # destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H1_1506_20190817_205359/' + file
    # if file.endswith('1506__0__20190817_205359_0_.txt'):
    #     shutil.move(original, destination)
    print(file)


for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H2BAR2_20210710_221000/' + file
    if file.endswith('BAR2_20210710_221000_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H4_1318_20150622_224558/' + file
    if file.endswith('1318__0__20150622_224558_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H4BAR4_20210711_234000/' + file
    if file.endswith('BAR4_20210711_234000_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H6_1318_20150620_212940/' + file
    if file.endswith('1318__0__20150620_212940_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H6_1506_20170528_004317/' + file
    if file.endswith('1506__0__20170528_004317_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/H6_1506_20190812_201720/' + file
    if file.endswith('1506__0__20190812_201720_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M1_1506_20150628_234925/' + file
    if file.endswith('1506__0__20150628_234925_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M1BAR1_20210721_010000/' + file
    if file.endswith('AAR1_20210721_010000_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M2_1506_20170512_220258/' + file
    if file.endswith('1506__0__20170512_220258_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M3_1318_20190823_220824/' + file
    if file.endswith('1318__0__20190823_220824_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M3_1324_20150702_215406/' + file
    if file.endswith('1324__0__20150702_215406_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M4BAR4_20210723_000000/' + file
    if file.endswith('BAR4_20210723_000000_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M5_1293_20190823_010403/' + file
    if file.endswith('1293__0__20190823_010403_0_.txt'):
        shutil.move(original, destination)

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    original = '/home/nottom/Documents/LinuxProject/test_data/text/background_backup/' + file
    destination = '/home/nottom/Documents/LinuxProject/test_data/text/trimm/M5_1324_20170512_222616/' + file
    if file.endswith('1324__0__20170512_222616_0_.txt'):
        shutil.move(original, destination)
