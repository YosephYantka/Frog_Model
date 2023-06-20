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

text_files = 'C:/pythonProject/PC/text_files'

#write each text file into a csv
for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])

    with open('C:/pythonProject/PC/text_files_for_csv/hopeful.csv', "w") as csv_file: #<destination for the csv
        writer = csv.writer(csv_file, delimiter=',')
        for line in list:
            writer.writerow(content)


#^this doesn't really work but maybe it will one day


list = [(0, 0, 0, 0), (0, 1, 1, 1)]

with open('csvfile.csv', 'wb') as file:
        file.write(content)