#this code is to sort spectrograms for a specific class into place
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

specgram_folder = '/home/nottom/Documents/LinuxProject/specgrams'

for file in os.listdir(specgram_folder):
    text_file = str(file[0:-7] + 'txt')

text_files = '/home/nottom/Documents/LinuxProject/text_files'

for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    if content == '1, 0, 0, 0':
        print(file)


fp = open(audacity_labels, "r")
data = fp.read()
print(data)

example = '/home/nottom/Documents/LinuxProject/text_files/6_10_BAR3_20210721_010000.txt'

fd = open(example, 'r')
datas = fd.read()
print(datas)