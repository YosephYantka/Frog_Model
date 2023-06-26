# this file is to create the text files for the spectrograms I've already sorted into the training data directories
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

# #STILL NEED TO DO TEXT FILES FOR 2015 AND 2017 TRAINING DATA -
# REMEMBER
# TO
# CHANGE
# WHERE
# THE
# BACKGROUND
# TEXTFILES
# ArE
# OUTPUTTING
# TO

# set file and folder(s)
audacity_labels = '/home/nottom/Documents/LinuxProject/audacity_labels/H6BAR5_20210707_030000.txt' #ALTER
chunk_file_raw = 'BAR5_20210707_030000.wav' #ALTER - make sure there are no characters after the time (eg. '...180000.wav'
chunk_folder = '/home/nottom/Documents/LinuxProject/chunks'
text_files = '/home/nottom/Documents/LinuxProject/text_files'
chunk_file = str(chunk_file_raw[0:-27] + '.wav')
# ^this file must be in the chunks folder
filename = str(chunk_file_raw[0:-4])
print(filename)


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)
        self.duration = math.ceil(self.audio.duration_seconds)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1 * 1000
        t2 = to_sec * 1 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")

    def multiple_split(self, sec_per_split):
        total_secs = math.ceil(self.get_duration())
        for i in range(0, total_secs, sec_per_split - 1):
            split_fn = str(i) + '_' + str(i + 4) + '_' + self.filename
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')

split_wav = SplitWavAudioMubin(chunk_folder, chunk_file_raw)

# for creating list of all annotations
fp = open(audacity_labels, "r")
data = fp.read()
mylist = data.replace('\n', ' ')
mylist = re.split('\t| ', mylist)
mylist = ([s for s in mylist if s != '\\'])
mylist = ([s for s in mylist if s != ''])
mylist = [float(x) for x in mylist]

# make a list of lists for annotations:
def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]

chunk_size = 5
annotations_list = list(split(mylist, chunk_size))

# create list of fours using length of recording
duration = split_wav.duration
segments = round(duration / 3)
fours = torch.linspace(4, duration, segments)

# create a corresponding dummy zeros tensor for the entire sound file
annotation = torch.zeros(segments, 4)

# Create a one-hot encoding tensor for each 4 second chunk
# THIS ONE WORKS (for some reason I have to write this into two pieces of code!!)
for t0, t1, id, f0, f1 in annotations_list:
    annotation[:, int(id)] = torch.logical_or(annotation[:, int(id)],
                                              torch.logical_and(fours - 0.5 > t0, fours - 3.5 < t1))
    annotation[:, 0] = torch.logical_and(fours > 0, fours > 0)
for t0, t1, id, f0, f1 in annotations_list:
    annotation[:, 0] = torch.logical_and(torch.sum(annotation, 1) == 0, fours > 0)

# write a text file for each 'chunk'
x = 0
y = 4
for chunk in annotation:
    name = str(x) + '_' + str(y) + '_' + filename
    encoding = chunk.tolist()
    encoding = [int(x) for x in encoding]
    encoding_string = f'{encoding[0]}, {encoding[1]}, {encoding[2]}, {encoding[3]}'
    with open('/home/nottom/Documents/LinuxProject/text_files/' + name + '.txt', 'x') as f:
        f.write(encoding_string)
    x = x + 3
    y = y + 3

#input a 0 for background chunks in text_files:
for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    if content == '0, 0, 0, 0':
        with open('/home/nottom/Documents/LinuxProject/text_files/' + file, 'w') as f:
            f.write(str('1, 0, 0, 0'))

#move text files into folders based on class:
for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])

    original = '/home/nottom/Documents/LinuxProject/text_files/' + filename + '.txt'
    destination = '/home/nottom/Documents/LinuxProject/training_data/text/notata/' + filename + '_1_.txt'
    if content == '0, 1, 0, 0':
        shutil.move(original, destination)

for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])

    original = '/home/nottom/Documents/LinuxProject/text_files/' + filename + '.txt'
    destination = '/home/nottom/Documents/LinuxProject/training_data/text/background_SELECTWHICHONE/' + filename + '_0_.txt'
    if content == '1, 0, 0, 0':
        shutil.move(original, destination)


#remove contents of text file
folder = '/home/nottom/Documents/LinuxProject/text_files/'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
folder = '/home/nottom/Documents/LinuxProject/chunks'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
