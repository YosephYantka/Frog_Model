import glob
import os
import shutil
import PIL
import random
import csv
from pathlib import Path
import glob

#test code
for file in glob.glob('/home/nottom/Documents/inference/test/**/*.wav', recursive=True):
    # print('length {}, file {}'.format( len(file), file[-12:]))
    # print(file)
    print(len(file))
    if len(file) == 133:
        os.renames(file, file[:-31] + '.wav')
    if len(file) == 85:
        os.renames(file, file[:-12] + '.wav')
    if len(file) == 119:
        os.renames(file, file[:-27] + '.wav')

#2021
for file in glob.glob('/home/nottom/Documents/inference/test/**/*.wav', recursive=True):


#BAR5
    if file[33:46] == 'BAR5_20210704':
        os.rename(file, file[:-4] + "_H6005_.wav")

    if file[33:46] == 'BAR5_20210705':
        os.rename(file, file[:-4] + "_H6005.wav")

    if file[33:46] == 'BAR5_20210706':
        os.rename(file, file[:-4] + "_H6070.wav")

    if file[33:46] == 'BAR5_20210707':
        os.rename(file, file[:-4] + "_H6070.wav")

    if file[33:46] == 'BAR5_20210710':
        os.rename(file, file[:-4] + "_H6170.wav")

    if file[33:46] == 'BAR5_20210711':
        os.rename(file, file[:-4] + "_H6170.wav")

#BAR4

    if file[33:46] == 'BAR4_20210705':
        os.rename(file, file[:-4] + "_H4170.wav")

    if file[33:46] == 'BAR4_20210706':
        os.rename(file, file[:-4] + "_H4070.wav")

    if file[33:46] == 'BAR4_20210707':
        os.rename(file, file[:-4] + "_H4070.wav")

    if file[33:46] == 'BAR4_20210708':
        os.rename(file, file[:-4] + "_H4005.wav")

    if file[33:46] == 'BAR4_20210709':
        os.rename(file, file[:-4] + "_H4005.wav")

    if file[33:46] == 'BAR4_20210710':
        os.rename(file, file[:-4] + "_H5005.wav")

    if file[33:46] == 'BAR4_20210711':
        os.rename(file, file[:-4] + "_H5005.wav")

    if file[33:46] == 'BAR4_20210712':
        os.rename(file, file[:-4] + "_H4170.wav")


#BAR3

    if file[33:46] == 'BAR3_20210704':
        os.rename(file, file[:-4] + "_H3170.wav")

    if file[33:46] == 'BAR3_20210705':
        os.rename(file, file[:-4] + "_H3170.wav")

    if file[33:46] == 'BAR3_20210706':
        os.rename(file, file[:-4] + "_H3070.wav")

    if file[33:46] == 'BAR3_20210707':
        os.rename(file, file[:-4] + "_H3070.wav")

    if file[33:46] == 'BAR3_20210708':
        os.rename(file, file[:-4] + "_H3005.wav")

    if file[33:46] == 'BAR3_20210709':
        os.rename(file, file[:-4] + "_H3005.wav")

    if file[33:46] == 'BAR3_20210711':
        os.rename(file, file[:-4] + "_H6005.wav")

    if file[33:46] == 'BAR3_20210712':
        os.rename(file, file[:-4] + "_H6005.wav")


# BAR2

    if file[33:46] == 'BAR2_20210705':
        os.rename(file, file[:-4] + "_H2070.wav")

    if file[33:46] == 'BAR2_20210706':
        os.rename(file, file[:-4] + "_H2170.wav")

    if file[33:46] == 'BAR2_20210707':
        os.rename(file, file[:-4] + "_H2170.wav")

    if file[33:46] == 'BAR2_20210708':
        os.rename(file, file[:-4] + "_H5170.wav")

    if file[33:46] == 'BAR2_20210709':
        os.rename(file, file[:-4] + "_H5170.wav")

    if file[33:46] == 'BAR2_20210710':
        os.rename(file, file[:-4] + "_H2005.wav")

    if file[33:46] == 'BAR2_20210711':
        os.rename(file, file[:-4] + "_H2005.wav")

    if file[33:46] == 'BAR2_20210712':
        os.rename(file, file[:-4] + "_H2070.wav")


# BAR1:

    if file[33:46] == 'AAR1_20210704':
        os.rename(file, file[:-4] + "_H5070.wav")

    if file[33:46] == 'AAR1_20210705':
        os.rename(file, file[:-4] + "_H5070.wav")

    if file[33:46] == 'AAR1_20210706':
        os.rename(file, file[:-4] + "_H1070.wav")

    if file[33:46] == 'AAR1_20210707':
        os.rename(file, file[:-4] + "_H1070.wav")

    if file[33:46] == 'AAR1_20210708':
        os.rename(file, file[:-4] + "_H1170.wav")

    if file[33:46] == 'AAR1_20210709':
        os.rename(file, file[:-4] + "_H1170.wav")

    if file[33:46] == 'AAR1_20210710':
        os.rename(file, file[:-4] + "_H1005.wav")

    if file[33:46] == 'AAR1_20210711':
        os.rename(file, file[:-4] + "_H1005.wav")
