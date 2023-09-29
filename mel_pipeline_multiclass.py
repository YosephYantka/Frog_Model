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
import shutil
import PIL
from tqdm import tqdm
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

matplotlib.use('qt5agg')
torch.random.manual_seed(0)

# set file and folder(s)
audacity_labels = '/home/nottom/Desktop/fake_labels_60_seconds.txt' #ALTER
chunk_file_raw = 'BAR4_20210723_080000.wav' #ALTER
chunk_folder = '/home/nottom/Documents/LinuxProject/chunks'
text_files = '/home/nottom/Documents/LinuxProject/text_files'
label_without_txt = audacity_labels[73:-4]
print(label_without_txt)
#chunk_file = str(chunk_file_raw[0:-27] + '.wav')
# ^this file must be in the chunks folder
filename = str(chunk_file_raw[0:-4])
print(filename)

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

split_wav = SplitWavAudioMubin(chunk_folder, chunk_file_raw)
split_wav.multiple_split(sec_per_split=4)
# os.remove('/home/nottom/Documents/LinuxProject/chunks/' + chunk_file_raw) # TODO: REMEMBER TO REACTIVATE THIS


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
    for i in tqdm(range(0, len(list_a), chunk_size), desc='splitting...'):
        yield list_a[i:i + chunk_size]

chunk_size = 5
annotations_list = list(split(mylist, chunk_size))

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
    encoding_string = f'{encoding[0]}, {encoding[1]}, {encoding[2]}, {encoding[3]}, {encoding[4]}, {encoding[5]}, {encoding[6]}, {encoding[7]}'
    with open('/home/nottom/Documents/LinuxProject/text_files/' + name + '.txt', 'x') as f:
        f.write(encoding_string)
    x +=  3
    y +=  3

#class for creating spectrogram
def save_spectrogram(specgram, title=None, ylabel="freq_bin"):
    # fig, axs = plt.subplots(1, 1)
    # axs.set_title(title or "Spectrogram (db)")
    # axs.set_ylabel(ylabel)
    # axs.set_xlabel("frame")
    # im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    # fig.colorbar(im, ax=axs)
    # plt.savefig("/home/nottom/Documents/LinuxProject/specgrams/" + str(title) + '.png')
    # plt.close()
    spec = librosa.power_to_db(specgram)
    plt.imsave("/home/nottom/Documents/LinuxProject/specgrams_raw/" + str(title) + '.png', spec)

#iterate through each file in the chunk folder and create a melspectrogram
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
    #save_spectrogram(spec[0], title= str(x) + '_' + str(y) + '_' + "spectrogram_" + str(filename))
    x = x + 3
    y = y + 3

#input a 1 for background chunks in text_files:
for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    if content == '0, 0, 0, 0, 0, 0, 0, 0':
        with open('/home/nottom/Documents/LinuxProject/training_data_2009/text/' + file, 'w') as f:
            f.write(str('1, 0, 0, 0, 0, 0, 0, 0'))

# This code will remove all segments that aren't of uniform size from spectrograms
folder = '/home/nottom/Documents/LinuxProject/specgrams'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    if file.startswith('3591'):
        os.unlink(join_path)
    if file[0:4] == '3594':
        os.unlink(join_path)

#this code will remove all segments that aren't of uniform size from text_files
folder = '/home/nottom/Documents/LinuxProject/text_files'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    if file.startswith('3591'):
        os.unlink(join_path)
    if file[0:4] == '3594':
        os.unlink(join_path)

#convert to greyscale:
folder = '/home/nottom/Documents/LinuxProject/specgrams_raw'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    image = PIL.Image.open(join_path).convert("L")
    image.save('/home/nottom/Documents/LinuxProject/specgrams/' + file)

#move spectrograms into folders based on class:
# class 0
for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])
    original = '/home/nottom/Documents/LinuxProject/specgrams/' + filename + '.wav.png'
    destination = '/home/nottom/Documents/LinuxProject/multi_class_model/training_data/specgrams/0/' + filename + '_0_.png'
    # print(content[6])
    if (content[0]) == '1':
        # print(file)
        shutil.move(original, destination)

for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])
    original = '/home/nottom/Documents/LinuxProject/training_data_2009/images/' + filename + '.wav.png'
    destination = '/home/nottom/Documents/LinuxProject/training_data_2009/specgrams/background/' + filename + '_0_.png'
    if content == '0, 0, 1, 0':
        shutil.move(original, destination)

#move text files into folders based on class:
for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])
    original = '/home/nottom/Documents/LinuxProject/training_data_2009/text/' + filename + '.txt'
    destination = '/home/nottom/Documents/LinuxProject/training_data_2009/text_files/notata/' + filename + '_1_.txt'
    if content == '0, 1, 0, 0':
        shutil.move(original, destination)

for file in os.listdir(text_files):
    join_path = os.path.join(text_files, file)
    f = open(join_path, 'r')
    content = f.read()
    filename = str(file[0:-4])
    original = '/home/nottom/Documents/LinuxProject/training_data_2009/text/' + filename + '.txt'
    destination = '/home/nottom/Documents/LinuxProject/training_data_2009/text_files/background/' + filename + '_0_.txt'
    if content == '0, 0, 1, 0':
        shutil.move(original, destination)

#clear contents of all folders that need clearing
folder = '/home/nottom/Documents/LinuxProject/training_data_2009/chunks'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# folder = '/home/nottom/Documents/LinuxProject/training_data_2009/text'
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))

# folder = '/home/nottom/Documents/LinuxProject/specgrams'
# for filename in os.listdir(folder):
#     file_path = os.path.join(folder, filename)
#     try:
#         if os.path.isfile(file_path) or os.path.islink(file_path):
#             os.unlink(file_path)
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#     except Exception as e:
#         print('Failed to delete %s. Reason: %s' % (file_path, e))

#clear contents of all folders that need clearing

folder = '/home/nottom/Documents/LinuxProject/training_data_2009/chunks'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
