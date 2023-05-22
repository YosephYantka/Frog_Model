import torch
import re
import csv
import wave
import contextlib
import numpy as np
import pprint
from pydub import AudioSegment
import math

#set file and folder(s)
chunk_folder = '/home/nottom/Documents/LinuxProject/BAR4_slices'
wav_file = 'BAR4_20210709_234000_Sunrise [-5.9183 142.6952].wav'
#^this file must be in the chunk_folder
filename = 'BAR4_20210709_234000_Sunrise [-5.9183 142.6952]'
audacity_labels = '/home/nottom/Documents/LinuxProject/labels.txt'

#Split audio file into 4 second clips
class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)

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
            split_fn = str(i) + '_' + str(i+4) + '_' + self.filename
            self.single_split(i, i + sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')

split_wav = SplitWavAudioMubin(chunk_folder, wav_file)
split_wav.multiple_split(sec_per_split=4)

# for creating list of all dummy annotations
fp = open(audacity_labels, "r")
data = fp.read()
mylist = data.replace('\n', ' ')
mylist = re.split('\t| ', mylist)
mylist = ([s for s in mylist if s != '\\'])
mylist = ([s for s in mylist if s != ''])
mylist = [float(x) for x in mylist]

#make a list of lists for annotations:
def split(list_a, chunk_size):
  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]
chunk_size = 5
annotations_list = list(split(mylist, chunk_size))

#create list of fours using length of recording
class GetAudioDuration():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def multiple_split(self):
        total_secs = math.ceil(self.get_duration())
        print(total_secs)
get_duration = GetAudioDuration(chunk_folder, wav_file)
get_duration.multiple_split()
#duration = get_duration.multiple_split() <- don't know how to assign the total_secs class variable to a variable outside of the class (duration)
duration = 3595
segments = round(3595/3)
fours = torch.linspace(4, duration,segments)

#create a corresponding dummy zeros tensor for the entire sound file
annotation = torch.zeros(segments, 4)

#Create a one-hot encoding tensor for each 4 second chunk
#THIS ONE WORKS (for some reason I have to write this into two pieces of code!!)
for t0,t1,id,f0,f1 in annotations_list:
    annotation[:, int(id)] = torch.logical_or(annotation[:, int(id)], torch.logical_and(fours > t0, fours - 4 < t1))
    annotation[:, 0] = torch.logical_and(fours > 0, fours > 0)
for t0,t1,id,f0,f1 in annotations_list:
    annotation[:, 0] = torch.logical_and(torch.sum(annotation, 1) == 0, fours > 0)

#write a text file for each 'chunk'
x = 0
y = 4
for chunk in annotation:
    name = 'Chunk_' + str(x) + '_' + str(y) + '_' + filename
    encoding = chunk.tolist()
    encoding = [int(x) for x in encoding]
    with open('/home/nottom/Documents/LinuxProject/BAR4_txt/' + name + '.txt', 'x') as f:
        f.write(str(encoding))
    x = x+3
    y = y+3

