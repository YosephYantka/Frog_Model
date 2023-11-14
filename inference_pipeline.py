# this code is for feeding project data through to the model
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
matplotlib.use('qt5agg')
torch.random.manual_seed(0)


# define functions/classes
class VGG16(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # changed the in_channels from 3 to 1 because my images are greyscale
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4 * 11 * 512, 4096),
            # will need to change the 7 * 7 because the feature maps are not going to be this size
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)  # .sigmoid()
        out = self.sigmoid(out)
        return out
class InferenceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        filename = self.img_labels.iloc[idx, 0]
        image = image.to(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, filename

def save_spectrogram(specgram, title=None, ylabel="freq_bin"):
    spec = librosa.power_to_db(specgram)
    plt.imsave("/home/nottom/Documents/inference/specgrams_raw/" + str(title) + '.png', spec)
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
def inference(model, device, inference_loader):
    model.load_state_dict(torch.load(
        "/home/nottom/Documents/LinuxProject/multi_class_model/multiclass_model.pt"))
    model.eval()
    with torch.no_grad():
        with open(chunk_file_raw + '_predictions.csv', 'w') as out_file:
            column_names = ['filename', 'prediction_0', 'prediction_1', 'prediction_2', 'prediction_3', 'prediction_4'
                            , 'prediction_5', 'prediction_6']
            writer = csv.DictWriter(out_file, fieldnames=column_names)
            writer.writeheader()
            for images, labels, filenames in inference_loader:
                images = images.to(device).type(torch.float) / 255

                # forward pass
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                # output a csv with filenames, predictions, thresholded predictions, elevation, distance, year
                for data in range(batch_size):
                    csv_out = csv.writer(out_file)
                    if outputs[data][0] > 0.9:
                        thresholded_0 = 1
                    else:
                        thresholded_0 = 0

                    if outputs[data][1] > 0.9:
                        thresholded_1 = 1
                    else:
                        thresholded_1 = 0

                    if outputs[data][2] > 0.9:
                        thresholded_2 = 1
                    else:
                        thresholded_2 = 0

                    if outputs[data][3] > 0.9:
                        thresholded_3 = 1
                    else:
                        thresholded_3 = 0

                    if outputs[data][4] > 0.9:
                        thresholded_4 = 1
                    else:
                        thresholded_4 = 0

                    if outputs[data][5] > 0.9:
                        thresholded_5 = 1
                    else:
                        thresholded_5 = 0

                    if outputs[data][6] > 0.9:
                        thresholded_6 = 1
                    else:
                        thresholded_6 = 0
                    csv_out.writerow([str(filenames[data]), thresholded_0, thresholded_1, thresholded_2
                                      , thresholded_3, thresholded_4, thresholded_5, thresholded_6])
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# set params
num_classes = 7
batch_size = 32
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.005

model = VGG16(num_classes).to(device)
model = model.apply(init_weights)

# set file and folder(s)
for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/**/*.wav', recursive=True):
    os.chdir('/home/nottom/Documents/inference')
    chunk_file_raw = file[-30:]
    print("next file: " + chunk_file_raw)
    chunk_folder = '/home/nottom/Documents/inference/chunks'

    # move recording to chunks folder:
    original = file
    destination = '/home/nottom/Documents/inference/chunks/' + chunk_file_raw
    shutil.copy(original, destination)

    split_wav = SplitWavAudioMubin(chunk_folder, chunk_file_raw)
    split_wav.multiple_split(sec_per_split=4)
    os.remove('/home/nottom/Documents/inference/chunks/' + chunk_file_raw)

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
    folder = '/home/nottom/Documents/inference/specgrams_raw'
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

    # convert to greyscale
    folder = '/home/nottom/Documents/inference/specgrams_raw'
    for file in os.listdir(folder):
        # print(file)
        join_path = os.path.join(folder, file)
        image = PIL.Image.open(join_path).convert("L")
        image.save('/home/nottom/Documents/inference/specgrams/' + file)

    folder = '/home/nottom/Documents/inference/specgrams'
    with open('annotations_file_binaryinference.csv', 'w') as out_file:
        csv_out = csv.writer(out_file)
        for file in os.listdir(folder):
            csv_out.writerow([file, 0])

    ## original = '/home/nottom/Documents/inference/specgrams/annotations_file_binaryinference.csv'
    ## destination = '/home/nottom/Documents/inference'

    inference_data = InferenceDataset(
        annotations_file='/home/nottom/Documents/inference/annotations_file_binaryinference.csv',
        img_dir='/home/nottom/Documents/inference/specgrams')
    inference_loader = DataLoader(inference_data, batch_size=batch_size, shuffle=False, drop_last=True)

    inference(model, 'cuda', inference_loader)

    import os
    folder = '/home/nottom/Documents/inference/chunks'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = '/home/nottom/Documents/inference/specgrams'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    folder = '/home/nottom/Documents/inference/specgrams_raw'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
