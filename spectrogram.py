import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset
import numpy
matplotlib.use('qt5agg')
torch.random.manual_seed(0)

SAMPLE_SPEECH = '/home/nottom/Documents/LinuxProject/BAR4_slices_1/BAR4_20210709_234000.wav'
filename = str(SAMPLE_SPEECH[-24:-1])

def save_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig("/home/nottom/Documents/LinuxProject/misc/" + str(title) + '.png')

#TO PLOT THE SPECTROGRAM:
SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(SAMPLE_SPEECH)
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
    power=2.0,)
# Perform transform
spec = spectrogram(SPEECH_WAVEFORM)
save_spectrogram(spec[0], title="Spectrogram_" + str(filename))

