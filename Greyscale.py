import PIL
import matplotlib
import os

#THIS CODE CONVERTS ALL RGBA FILES TO GREYSCALE
folder = '/home/nottom/Documents/LinuxProject/training_data_2009/1318_20170515_231921/specgrams/background'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    image = PIL.Image.open(join_path).convert("L")
    image.save('/home/nottom/Documents/LinuxProject/training_data_2009/1318_20170515_231921/specgrams_real/background/' + file)

folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_training'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    image = PIL.Image.open(join_path).convert("L")
    image.save('/home/nottom/Documents/LinuxProject/first_model/img_dir_training_grey/' + file)


import os
import PIL
import shutil
folder = '/home/nottom/Documents/LinuxProject/first_model/backups/img_dir_training'
for file in folder:
    if file.endswith('_0_.png'):
        with open('/home/nottom/Documents/LinuxProject/first_model/balanced_text_training/' + file[:-4] + '.txt', 'x') as f:
            f.write("1")
    if file.endswith('_1_.png'):
        with open('/home/nottom/Documents/LinuxProject/first_model/balanced_text_training/' + file[:-4] + '.txt', 'x') as f:
            f.write("2")

#For creating balanced datasets - culls total background directory and creates a corresponding text directory
import shutil
import PIL
import os
#culling
folder = '/home/nottom/Documents/LinuxProject/first_model/backups/img_dir_training'
# folder = '/home/nottom/Documents/LinuxProject/first_model/backups/training_data_text'
from random import sample
#creates new corresponding text file
files = os.listdir('/home/nottom/Documents/LinuxProject/first_model/backups/img_dir_training')
for file in sample(files,12745):
    path = os.path.join(folder, file)
    os.unlink(path)
