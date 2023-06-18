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
import shutil

chunker = '/home/nottom/Documents/LinuxProject/chunks'

#USE THE EXTENSION SHUTIL FOR MOVING FILES AND DIRECTORIES!!
#HAVE TO SWAP THE BLACK SLASHES FOR FORWARD SLASHES WHEN COPYING DIRECTORY PATH!!
#https://stackoverflow.com/questions/69428860/move-files-from-a-directory-to-another-based-on-a-condition

file = '0_4_H6BAR5_20190813_231720.txt'

text_files = '/home/nottom/Documents/LinuxProject/text_files'

for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])

    original = '/home/nottom/Documents/LinuxProject/specgrams/' + filename + '.wav.png'
    destination = '/home/nottom/Documents/LinuxProject/training_data/notata/' + filename + '_1_.png'
    if content == '0, 1, 0, 0':
        shutil.move(original, destination)

for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])

    original = '/home/nottom/Documents/LinuxProject/specgrams/' + filename + '.wav.png'
    destination = '/home/nottom/Documents/LinuxProject/training_data/background/' + filename + '_0_.png'
    if content == '1, 0, 0, 0':
        shutil.move(original, destination)

for file in os.listdir(chunker):
    join_path = os.path.join(chunker, file)
    f = open(join_path, 'r')
    os.remove(file)

import os, shutil
folder = chunker
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

