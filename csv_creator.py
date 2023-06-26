#this file is for creating a csv out of the text files in a directory for training
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import re
from pydub import AudioSegment
import math
import matplotlib
import csv
import librosa
import matplotlib.pyplot as plt
import os
import shutil

text_files = '/home/nottom/Documents/LinuxProject/training_data/text/background_2019_2021'

os.chdir('/home/nottom/Documents/LinuxProject/training_data/text/background_2019_2021')

from pathlib import Path
with open('big.csv', 'w') as out_file:
    csv_out = csv.writer(out_file)
    csv_out.writerow(['FileName', 'Content'])
    for fileName in Path('.').glob('*.txt'):
        csv_out.writerow([str(fileName),open(str(fileName.absolute())).read().strip()])





#
# #write each text file into a csv
# for file in os.listdir(text_files):
#     join_path = os.path.join(text_files, file)
#     f = open(join_path, 'r')
#     content = f.read()
#     filename = str(file[0:-4])
#
#     with open('/home/nottom/Documents/LinuxProject/training_data/text/hopeful.csv', "w") as csv_file: #<destination for the csv
#         writer = csv.writer(csv_file, delimiter=',')
#         for line in list:
#             writer.writerow(content)
#
#
# #^this doesn't really work but maybe it will one day
#
# import glob
# os.chdir('/home/nottom/Documents/LinuxProject/training_data/text/background_2019_2021')
# glob.glob("*.txt")
# globlet = glob.glob("*.txt")
# print(globlet)
#
#
# csv_list = []
# for file in os.listdir(text_files):
#     join_path = os.path.join(text_files, file)
#     f = open(join_path, 'r')
#     content = f.read()
#     splitlist = content.split(', ')
#     splitlist = [int(x) for x in splitlist]
#     csv_list.append(splitlist)
# #
# print(csv_list)
# type(csv_list[2][2])
# #
# # def split(list_a, chunk_size):
# #     for i in range(0, len(list_a), chunk_size):
# #         yield list_a[i:i + chunk_size]
# #
# # chunk_size = 1
# # csv_list_oflists = list(split(csv_list, chunk_size))
# # print(csv_list_oflists)
#
# with open("/home/nottom/Documents/LinuxProject/training_data/text/hopeful.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(csv_list)
#
# #new one
#
#
