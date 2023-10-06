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
for file in glob.glob('/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav', recursive=True):
    x = file[62:75]
    # print(x)
    if (file[0:46]) == '/media/nottom/TOSHIBA EXT/2021/Hides/BAR5/BAR5':
        # print(file[46:])

        # print(file[47:55])
    if file[62:75] == 'BAR5_20210709':

    if file[62:75] == 'BAR5_20210708':

    if file[62:75] == 'BAR5_20210710'

    if file[62:75] == 'BAR5_20210706'

    if file[62:75] == 'AAR1_20210704'

    if file[62:75] == 'AAR1_20210705'

    if file[62:75] == 'AAR1_20210706'

    if file[62:75] == 'AAR1_20210707'

    if file[62:75] == 'AAR1_20210708'

    if file[62:75] == 'AAR1_20210709'

    if file[62:75] == 'AAR1_20210710'

    if file[62:75] == 'AAR1_20210711'

    if file[62:75] == 'AAR1_20210712'
    #
    if file[62:75] == 'AAR1_20210707'

    if file[62:75] == 'AAR1_20210708'

    if file[62:75] == 'AAR1_20210709'

    if file[62:75] == 'AAR1_20210710'

    if file[62:75] == 'AAR1_20210711'

    if file[62:75] == 'AAR1_20210712'


#code to move all files into a single directory
for file in glob.glob('/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav', recursive=True):
    original = file
    destination = '' #<- to be decided!
    shutil.move(original, destination)
