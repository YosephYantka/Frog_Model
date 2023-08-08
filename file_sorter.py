# this file chunks specified wav files, and creates and exports text documents and spectrograms for each chunk
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import re
from pydub import AudioSegment
import math
import matplotlib
import librosa
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil



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
