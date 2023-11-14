# this code is for creating training data from labels
import csv
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import re
import glob
from pydub import AudioSegment
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib
import librosa
import matplotlib.pyplot as plt
import os
import shutil
import PIL
from torchvision.io import read_image
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# define functions/classes
def save_spectrogram(specgram, title=None, ylabel="freq_bin"):
    spec = librosa.power_to_db(specgram)
    plt.imsave("/home/nottom/Documents/Training/specgrams_raw/" + str(title) + '.png', spec)
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
            # print(str(i) + ' Done')
            # if i == total_secs - sec_per_split:
            #     # print('All splited successfully')
def split(list_a, chunk_size):
    for i in tqdm(range(0, len(list_a), chunk_size), desc='splitting...'):
        yield list_a[i:i + chunk_size]


# set file and folder(s)
for file in glob.glob('/media/nottom/TOSHIBA EXT/training_labels_and_recordings/labels/multi/**/*.txt', recursive=True):
    os.chdir('/home/nottom/Documents/Training')
    chunk_file_raw = file[-30:-4] + '.wav'
    chunk_folder = '/home/nottom/Documents/Training/chunks'
    text_files = '/home/nottom/Documents/Training/text_files'
    filename = str(chunk_file_raw[0:-4])

    # move recording to chunks folder:
    original = '/media/nottom/TOSHIBA EXT/training_labels_and_recordings/recordings/' + chunk_file_raw
    print(original)
    destination = '/home/nottom/Documents/Training/chunks/' + chunk_file_raw
    shutil.copy(original, destination)

    split_wav = SplitWavAudioMubin(chunk_folder, chunk_file_raw)
    split_wav.multiple_split(sec_per_split=4)
    os.remove('/home/nottom/Documents/Training/chunks/' + chunk_file_raw)

    # for creating list of all annotations
    fp = open(file, "r")
    data = fp.read()
    mylist = data.replace('\n', ' ')
    mylist = re.split('\t| ', mylist)
    mylist = ([s for s in mylist if s != '\\'])
    mylist = ([s for s in mylist if s != ''])
    mylist = [float(x) for x in mylist]

    # make a list of lists for annotations:
    def split(list_a, chunk_size):
        for i in tqdm(range(0, len(list_a), chunk_size), desc='splitting...'):
            yield list_a[i:i + chunk_size]

    chunk_size = 5
    annotations_list = list(split(mylist, chunk_size))
    for x in annotations_list:
        if x[2] == 0:
            x[2] = 1.0

    # create list of fours using length of recording
    duration = split_wav.duration
    segments = round(duration / 3)
    fours = torch.linspace(4, duration, segments)

    # create a corresponding dummy zeros tensor for the entire sound file
    annotation = torch.zeros(segments, 8)
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
        encoding_string = f'{encoding[7]}, {encoding[1]}, {encoding[2]}, {encoding[3]}, {encoding[4]}, {encoding[5]}, {encoding[6]}'
        with open('/home/nottom/Documents/Training/text_files/' + name + '.txt', 'x') as f:
            f.write(encoding_string)
        x += 3
        y += 3

    # creates melspectrograms
    directory = chunk_folder
    x = 0
    y = 4
    for file in os.listdir(chunk_folder):
        f = os.path.join(directory, file)
        SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(f)
        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128
        sample_rate = 48000
        # Define transform
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk",
        )
        # Perform transform
        melspec = mel_spectrogram(SPEECH_WAVEFORM)
        save_spectrogram(melspec[0], title=str(file))
        x = x + 3
        y = y + 3

    # This code will remove all segments that aren't of uniform size from specgrams_raw
    folder = '/home/nottom/Documents/Training/specgrams_raw'
    for file in os.listdir(folder):
        join_path = os.path.join(folder, file)
        if file.startswith('3591'):
            os.unlink(join_path)
        if file[0:4] == '3594':
            os.unlink(join_path)

    for file in os.listdir(folder):
        join_path = os.path.join(folder, file)
        img = PIL.Image.open(join_path)
        width = img.width
        height = img.height
        # print(height, width)
        if width != 376:
            os.unlink(join_path)

    folder = '/home/nottom/Documents/Training/text_files'
    for file in os.listdir(folder):
        join_path = os.path.join(folder, file)
        if file.startswith('3591'):
            os.unlink(join_path)
        if file[0:4] == '3594':
            os.unlink(join_path)

    # convert to greyscale
    folder = '/home/nottom/Documents/Training/specgrams_raw'
    for file in os.listdir(folder):
        print(file)
        join_path = os.path.join(folder, file)
        image = PIL.Image.open(join_path).convert("L")
        image.save('/home/nottom/Documents/Training/specgrams/' + file)

    # # move spectrograms into folders based on class:
    # # class 0
    # for file in os.listdir(text_files):
    #     join_path = os.path.join(text_files, file)
    #     f = open(join_path, 'r')
    #     content = f.read()
    #     filename = str(file[0:-4])
    #     original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
    #     destination = '/home/nottom/Documents/Training/training_data/0/specgrams/' + filename + '_0_.png'
    #     if (content[0]) == '1':
    #         shutil.move(original, destination)
    #     original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
    #     destination = '/home/nottom/Documents/Training/training_data/0/text/' + filename + '_0_.png.txt'
    #     if (content[0]) == '1':
    #         shutil.move(original, destination)

    # class 1
    for file in os.listdir(text_files):
        join_path = os.path.join(text_files, file)
        f = open(join_path, 'r')
        content = f.read()
        filename = str(file[0:-4])
        original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/specgrams/' + filename + '_1_.png'
        if (content[3]) == '1':
            shutil.move(original, destination)
        original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/text/' + filename + '_1_.png.txt'
        if (content[3]) == '1':
            shutil.move(original, destination)

    # class 2
    for file in os.listdir(text_files):
        join_path = os.path.join(text_files, file)
        f = open(join_path, 'r')
        content = f.read()
        filename = str(file[0:-4])
        original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/specgrams/' + filename + '_2_.png'
        if (content[6]) == '1':
            shutil.move(original, destination)
        original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/text/' + filename + '_2_.png.txt'
        if (content[6]) == '1':
            shutil.move(original, destination)

    # class 3
    for file in os.listdir(text_files):
        join_path = os.path.join(text_files, file)
        f = open(join_path, 'r')
        content = f.read()
        filename = str(file[0:-4])
        original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/specgrams/' + filename + '_3_.png'
        if (content[9]) == '1':
            shutil.move(original, destination)
        original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/text/' + filename + '_3_.png.txt'
        if (content[9]) == '1':
            shutil.move(original, destination)
    # class 4
    for file in os.listdir(text_files):
        join_path = os.path.join(text_files, file)
        f = open(join_path, 'r')
        content = f.read()
        filename = str(file[0:-4])
        original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/specgrams/' + filename + '_4_.png'
        if (content[12]) == '1':
            shutil.move(original, destination)
        original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/text/' + filename + '_4_.png.txt'
        if (content[12]) == '1':
            shutil.move(original, destination)
    # class 5
    for file in os.listdir(text_files):
        join_path = os.path.join(text_files, file)
        f = open(join_path, 'r')
        content = f.read()
        filename = str(file[0:-4])
        original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/specgrams/' + filename + '_5_.png'
        if (content[15]) == '1':
            shutil.move(original, destination)
        original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/text/' + filename + '_5_.png.txt'
        if (content[15]) == '1':
            shutil.move(original, destination)
    # class 6
    for file in os.listdir(text_files):
        join_path = os.path.join(text_files, file)
        f = open(join_path, 'r')
        content = f.read()
        filename = str(file[0:-4])
        original = '/home/nottom/Documents/Training/specgrams/' + filename + '.wav.png'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/specgrams/' + filename + '_6_.png'
        if (content[18]) == '1':
            shutil.move(original, destination)
        original = '/home/nottom/Documents/Training/text_files/' + filename + '.txt'
        destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/new_ones/text' + filename + '_6_.png.txt'
        if (content[18]) == '1':
            shutil.move(original, destination)

    #empties directories
    import os
    folder = '/home/nottom/Documents/Training/chunks'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = '/home/nottom/Documents/Training/specgrams'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = '/home/nottom/Documents/Training/specgrams_raw'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = '/home/nottom/Documents/Training/text_files'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    print('finished!')
