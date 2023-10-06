import glob
import os
import shutil
import PIL
import random
import csv
from pathlib import Path
import glob

folder = '/media/nottom/TOSHIBA EXT/2021/Hides/BAR1/AAR1_20210704_STUDY'
os.chdir('/media/nottom/TOSHIBA EXT/2021/Hides/BAR1/AAR1_20210704_STUDY')


#code for chopping off the 'sunrise' and coordinates from files!
for file in glob.glob('/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav', recursive=True):
    # print('length {}, file {}'.format( len(file), file[-12:]))
    if len(file) == 132:
        os.renames(file, file[:-31] + '.wav')
    if len(file) == 94:
        os.renames(file, file[:-12] + '.wav')
    if len(file) == 128:
        os.renames(file, file[:-27] + '.wav')


#code for renaming 2021 files to include metadata
#todo: TRIAL THIS CODE ON A SMALL AMOUNT OF DATA FIRST!!!!
for file in glob.glob('/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav', recursive=True):
    print(file[42:55])
    if (file[0:46]) == '/media/nottom/TOSHIBA EXT/2021/Hides/BAR4/BAR4':
        print(file[46:])

#BAR5
    if file[42:55] == 'BAR5_20210705':

    if file[42:55] == 'BAR5_20210706':

    if file[42:55] == 'BAR5_20210707'

    if file[42:55] == 'BAR5_20210708'

    if file[42:55] == 'BAR5_20210709':

    if file[42:55] == 'BAR5_20210710':

    if file[42:55] == 'BAR5_20210711'

    if file[42:55] == 'BAR5_20210712'

#BAR4

    if file[42:55] == 'BAR4_20210705':

    if file[42:55] == 'BAR4_20210706':

    if file[42:55] == 'BAR4_20210707'

    if file[42:55] == 'BAR4_20210708'

    if file[42:55] == 'BAR4_20210709':

    if file[42:55] == 'BAR4_20210710':

    if file[42:55] == 'BAR4_20210711'

    if file[42:55] == 'BAR4_20210712'

    if file[42:55] == 'BAR4_20210713'

#BAR3

    if file[42:55] == 'BAR3_20210704':

    if file[42:55] == 'BAR3_20210705':

    if file[42:55] == 'BAR3_20210706':

    if file[42:55] == 'BAR3_20210707'

    if file[42:55] == 'BAR3_20210708'

    if file[42:55] == 'BAR3_20210709':

    if file[42:55] == 'BAR3_20210710':

    if file[42:55] == 'BAR3_20210711'

    if file[42:55] == 'BAR3_20210712'

    if file[42:55] == 'BAR3_20210713'

# BAR2

    if file[42:55] == 'BAR2_20210705':

    if file[42:55] == 'BAR2_20210706':

    if file[42:55] == 'BAR2_20210707'

    if file[42:55] == 'BAR2_20210708'

    if file[42:55] == 'BAR2_20210709':

    if file[42:55] == 'BAR2_20210710':

    if file[42:55] == 'BAR2_20210711'

    if file[42:55] == 'BAR2_20210712'

    if file[42:55] == 'BAR2_20210713'

# BAR1:

    if file[42:55] == 'AAR1_20210704':

    if file[42:55] == 'AAR1_20210705':

    if file[42:55] == 'AAR1_20210706':

    if file[42:55] == 'AAR1_20210707'

    if file[42:55] == 'AAR1_20210708'

    if file[42:55] == 'AAR1_20210709':

    if file[42:55] == 'AAR1_20210710':

    if file[42:55] == 'AAR1_20210711'

    if file[42:55] == 'AAR1_20210712'

#code to move all files into a single directory
for file in glob.glob('/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav', recursive=True):
    original = file
    destination = '' #<- to be decided!
    shutil.move(original, destination)
