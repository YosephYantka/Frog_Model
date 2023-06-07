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

matplotlib.use('qt5agg')
torch.random.manual_seed(0)

# set file and folder(s)
chunk_folder = '/home/nottom/Documents/LinuxProject/BAR4_slices'
chunk_file = 'BAR4_20210709_234000.wav'
# ^this file must be in the chunk_folder
#wav_file = '/home/nottom/Documents/LinuxProject/BAR4_slices_1/BAR4_20210709_234000.wav'
filename = str(chunk_file[0:-4])
audacity_labels = '/home/nottom/Documents/LinuxProject/labels.txt'

# Split audio file into 4 second clips
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

split_wav = SplitWavAudioMubin(chunk_folder, chunk_file)
split_wav.multiple_split(sec_per_split=4)

# for creating list of all dummy annotations
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
    with open('/home/nottom/Documents/LinuxProject/BAR4_txt/' + name + '.txt', 'x') as f:
        f.write(encoding_string)
    x = x + 3
    y = y + 3

#class for creating spectrogram

def save_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig("/home/nottom/Documents/LinuxProject/BAR4_spectrograms/" + str(title) + '.png')
    plt.close()

#iterate through each file in the chunk folder
directory = chunk_folder
x = 0
y = 4
for file in os.listdir(chunk_folder):
    f = os.path.join(directory, file)
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(f)
    n_fft = 1024
    win_length = None
    hop_length = 512
    # Define transform
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0, )
    # Perform transform
    spec = spectrogram(SPEECH_WAVEFORM)
    save_spectrogram(spec[0], title=str(file))
    #save_spectrogram(spec[0], title= str(x) + '_' + str(y) + '_' + "spectrogram_" + str(filename))
    x = x + 3
    y = y + 3

