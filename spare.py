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

folder = '/home/nottom/Documents/LinuxProject/training_data/spectrograms'

# use this to convert all rgb images to greyscale
import PIL
import matplotlib
import os
folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_valid'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    image = PIL.Image.open(join_path).convert("L")
    image.save('/home/nottom/Documents/LinuxProject/first_model/img_dir_valid_grey/' + file)

folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_training'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    image = PIL.Image.open(join_path).convert("L")
    image.save('/home/nottom/Documents/LinuxProject/first_model/img_dir_training_grey/' + file)


# use this to check if all images are the same size (THEY AREN'T)
folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_training_grey'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    img = PIL.Image.open(join_path)
    width = img.width
    height = img.height
    # print(height, width)

    if width != 376:
        print(file)
    # if height != 128:
    #     print(file)

looloo = '/home/nottom/Documents/LinuxProject/miscandfashionmnist/54_58_BAR2_20210710_221000_0_.png'
looloo.size()

# check is pad_sequence works
