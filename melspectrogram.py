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

# (Path.cwd() / 'data' / 'stuff').mkdir(parents=True, exist_ok=True)
# sorted((Path.cwd() / 'data').glob('*.png'))
#
# Path('home/otherstuff.png').stem
SAMPLE_SPEECH = '/home/nottom/Documents/LinuxProject/chunks/21_25_H6BAR5_20190813_231720.wav'
SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)

def save_spectrogram_mel(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig("/home/nottom/Documents/LinuxProject/one/" + 'mel.png')
    plt.close()

def save_spectrogram_normal(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig("/home/nottom/Documents/LinuxProject/one/" + 'normal.png')
    plt.close()

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
sample_rate = 16000
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)
melspec = mel_spectrogram(SPEECH_WAVEFORM)
save_spectrogram_mel(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")


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
    power=2.0,
)
# Perform transform
spec = spectrogram(SPEECH_WAVEFORM)
save_spectrogram_normal(spec[0], title="torchaudio")

folder = '/home/nottom/Documents/LinuxProject/training_data/text/background_2019_2021'

lala = '318_322_AAR1_20210708_184000_0_.txt'
# print(lala[10:30])

for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    if file.endswith('AAR1_20210708_184000_0_.txt'):
        os.unlink(join_path)
    elif os.path.isdir(join_path):
        shutil.rmtree(join_path)


    if file[11:30] == 'BAR4_20210718_140000':
        os. unlink(file)
