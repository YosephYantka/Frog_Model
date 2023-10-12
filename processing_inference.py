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


#/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav
# #code for chopping off the 'sunrise' and coordinates from files!
# for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/Hides 2015/**/*.wav', recursive=True):
#     # print('length {}, file {}'.format( len(file), file))
#     if len(file) != 86:# and len(file) != 86:
#          print(file)
# # '/media/nottom/TOSHIBA EXT/joseph_dataset/Hides 2015/1324/Data/1324_20150612_153308.wav'
#     if file[62:7-] != '1324_201':
#         print(file)
#         newname = file.replace('_0+1_','')
#         os.renames(file, newname)
#         os.renames(file, file[:-31] + '.wav')
#     if len(file) == 109:
#         os.renames(file, file[:-12] + '.wav')
#     if len(file) == 143:
#         os.renames(file, file[:-27] + '.wav')

#test code #todo run this code on each year individually in case something goes wrong
for file in glob.glob('/home/nottom/Documents/test/**/*.wav', recursive=True):
    # print('length {}, file {}'.format( len(file), file[-12:]))
    print(file)
    if len(file) == 123:
        os.renames(file, file[:-31] + '.wav')
    if len(file) == 85:
        os.renames(file, file[:-12] + '.wav')
    if len(file) == 119:
        os.renames(file, file[:-27] + '.wav')

#2021
#todo  CONFIRM WHETHER ALL THE NIGHTS ARE CORRECT!! REMOVE FILES FROM JJ.DATASET THAT AREN'T IN SPREADHSEET
for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/Hides 2021/**/*.wav', recursive=True):
    # print(file)
    # if (file[0:46]) == '/media/nottom/TOSHIBA EXT/2021/Hides/BAR4/BAR4':
    # print(file[57:70])

    # # BAR5
    # if file[57:70] == 'BAR5_20210704':
    #     os.rename(file, file[:-4] + "_H6005_.wav")
    #
    # if file[57:70] == 'BAR5_20210705':
    #     os.rename(file, file[:-4] + "_H6005.wav")
    #
    # if file[57:70] == 'BAR5_20210706':
    #     os.rename(file, file[:-4] + "_H6070.wav")
    #
    # if file[57:70] == 'BAR5_20210707':
    #     os.rename(file, file[:-4] + "_H6070.wav")
    #
    # if file[57:70] == 'BAR5_20210710':
    #     os.rename(file, file[:-4] + "_H6170.wav")
    #
    # if file[57:70] == 'BAR5_20210711':
    #     os.rename(file, file[:-4] + "_H6170.wav")

    # BAR4
    # if file[57:70] == 'BAR4_20210705':
    #     os.rename(file, file[:-4] + "_H4170.wav")
    #
    # if file[57:70] == 'BAR4_20210706':
    #     os.rename(file, file[:-4] + "_H4070.wav")
    #
    # if file[57:70] == 'BAR4_20210707':
    #     os.rename(file, file[:-4] + "_H4070.wav")
    #
    # if file[57:70] == 'BAR4_20210708':
    #     os.rename(file, file[:-4] + "_H4005.wav")
    #
    # if file[57:70] == 'BAR4_20210709':
    #     os.rename(file, file[:-4] + "_H4005.wav")
    #
    # if file[57:70] == 'BAR4_20210710':
    #     os.rename(file, file[:-4] + "_H5005.wav")
    #
    # if file[57:70] == 'BAR4_20210711':
    #     os.rename(file, file[:-4] + "_H5005.wav")
    #
    # if file[57:70] == 'BAR4_20210712':
    #     os.rename(file, file[:-4] + "_H4170.wav")

    # BAR3
    #
    # if file[57:70] == 'BAR3_20210704':
    #     os.rename(file, file[:-4] + "_H3170.wav")
    #
    # if file[57:70] == 'BAR3_20210705':
    #     os.rename(file, file[:-4] + "_H3170.wav")
    #
    # if file[57:70] == 'BAR3_20210706':
    #     os.rename(file, file[:-4] + "_H3070.wav")
    #
    # if file[57:70] == 'BAR3_20210707':
    #     os.rename(file, file[:-4] + "_H3070.wav")
    #
    # if file[57:70] == 'BAR3_20210708':
    #     os.rename(file, file[:-4] + "_H3005.wav")
    #
    # if file[57:70] == 'BAR3_20210709':
    #     os.rename(file, file[:-4] + "_H3005.wav")
    #
    # if file[57:70] == 'BAR3_20210711':
    #     os.rename(file, file[:-4] + "_H6005.wav")
    #
    # if file[57:70] == 'BAR3_20210712':
    #     os.rename(file, file[:-4] + "_H6005.wav")

    # # BAR2
    #
    # if file[57:70] == 'BAR2_20210705':
    #     os.rename(file, file[:-4] + "_H2070.wav")
    #
    # if file[57:70] == 'BAR2_20210706':
    #     os.rename(file, file[:-4] + "_H2170.wav")
    #
    # if file[57:70] == 'BAR2_20210707':
    #     os.rename(file, file[:-4] + "_H2170.wav")
    #
    # if file[57:70] == 'BAR2_20210708':
    #     os.rename(file, file[:-4] + "_H5170.wav")
    #
    # if file[57:70] == 'BAR2_20210709':
    #     os.rename(file, file[:-4] + "_H5170.wav")
    #
    # if file[57:70] == 'BAR2_20210710':
    #     os.rename(file, file[:-4] + "_H2005.wav")
    #
    # if file[57:70] == 'BAR2_20210711':
    #     os.rename(file, file[:-4] + "_H2005.wav")
    #
    # if file[57:70] == 'BAR2_20210712':
    #     os.rename(file, file[:-4] + "_H2070.wav")

    # BAR1:
    #
    # if file[57:70] == 'AAR1_20210704':
    #     os.rename(file, file[:-4] + "_H5070.wav")
    #
    # if file[57:70] == 'AAR1_20210705':
    #     os.rename(file, file[:-4] + "_H5070.wav")
    #
    # if file[57:70] == 'AAR1_20210706':
    #     os.rename(file, file[:-4] + "_H1070.wav")
    #
    # if file[57:70] == 'AAR1_20210707':
    #     os.rename(file, file[:-4] + "_H1070.wav")
    #
    # if file[57:70] == 'AAR1_20210708':
    #     os.rename(file, file[:-4] + "_H1170.wav")
    #
    # if file[57:70] == 'AAR1_20210709':
    #     os.rename(file, file[:-4] + "_H1170.wav")
    #
    # if file[57:70] == 'AAR1_20210710':
    #     os.rename(file, file[:-4] + "_H1005.wav")
    #
    # if file[57:70] == 'AAR1_20210711':
    #     os.rename(file, file[:-4] + "_H1005.wav")

#2019  #todo 1193A does not have a data file so it will need different lengths
for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/Hides 2019/**/*.wav', recursive=True):
        print(file)
    # if len(file) == 87:
    #     print(file[63:76])
    # if file[58:71] == '1193_20190811':
    #     os.rename(file, file[:-4] + "_H5070")
    #
    # if file[58:71] == '1193_20190812':
    #     os.rename(file, file[:-4] + "_H5070")
    #
    # if file[58:71] == '1193_20190813':
    #     os.rename(file, file[:-4] + "_H5170")
    #
    # if file[58:71] == '1193_20190814':
    #     os.rename(file, file[:-4] + "_H5170")
    #
    # if file[58:71] == '1193_20190815':
    #     os.rename(file, file[:-4] + "_H4005")
    #
    # if file[58:71] == '1193_20190816':
    #     os.rename(file, file[:-4] + "_H4005")

#1293
    # if file[63:76] == '1293_20190810':
    #     os.rename(file, file[:-4] + "_H5005.wav")
#
#     if file[63:76] == '1293_20190811':
#         os.rename(file, file[:-4] + "_H5005.wav")
#
#     if file[63:76] == '1293_20190812':
#         os.rename(file, file[:-4] + "_H2005")
#
#     if file[63:76] == '1293_20190813':
#         os.rename(file, file[:-4] + "_H2005")
#
#     if file[63:76] == '1293_20190814':
#         os.rename(file, file[:-4] + "_H2070")
#
#     if file[63:76] == '1293_20190815':
#         os.rename(file, file[:-4] + "_H2070")
#
#     if file[63:76] == '1293_20190816':
#         os.rename(file, file[:-4] + "_H4170")
#
#     if file[63:76] == '1293_20190817':
#         os.rename(file, file[:-4] + "_H4170")
#
# #1318
#     if file[63:76] == '1318_20190810':
#         os.rename(file, file[:-4] + "_H6070")
#
#     if file[63:76] == '1318_20190811':
#         os.rename(file, file[:-4] + "_H6070")
#
#     if file[63:76] == '1318_20190812':
#         os.rename(file, file[:-4] + "_H3070")
#
#     if file[63:76] == '1318_20190813':
#         os.rename(file, file[:-4] + "_H3070")
#
#     if file[63:76] == '1318_20190814':
#         os.rename(file, file[:-4] + "_H2170")
#
#     if file[63:76] == '1318_20190815':
#         os.rename(file, file[:-4] + "_H2170")
#
#     if file[63:76] == '1318_20190816':
#         os.rename(file, file[:-4] + "_H4070")
#
#     if file[63:76] == '1318_20190817':
#         os.rename(file, file[:-4] + "_H4070")
#
# #1324
#     if file[63:76] == '1324_20190811':
#         os.rename(file, file[:-4] + "_H3005")
#
#     if file[63:76] == '1324_20190812':
#         os.rename(file, file[:-4] + "_H3005")
#
#     if file[63:76] == '1324_20190813':
#         os.rename(file, file[:-4] + "_H3170")
#
#     if file[63:76] == '1324_20190814':
#         os.rename(file, file[:-4] + "_H3170")
#
#     if file[63:76] == '1324_20190816':
#         os.rename(file, file[:-4] + "_H1070")
#
#     if file[63:76] == '1324_20190817':
#         os.rename(file, file[:-4] + "_H1070")
#
# #1506
#     if file[63:76] == '1506_20190810':
#         os.rename(file, file[:-4] + "_H6005")
#
#     if file[63:76] == '1506_20190811':
#         os.rename(file, file[:-4] + "_H6005")
#
#     if file[63:76] == '1506_20190812':
#         os.rename(file, file[:-4] + "_H6170")
#
#     if file[63:76] == '1506_20190813':
#         os.rename(file, file[:-4] + "_H6170")
#
#     if file[63:76] == '1506_20190814':
#         os.rename(file, file[:-4] + "_H1005")
#
#     if file[63:76] == '1506_20190815':
#         os.rename(file, file[:-4] + "_H1005")
#
#     if file[63:76] == '1506_20190816':
#         os.rename(file, file[:-4] + "_H1170")
#
#     if file[63:76] == '1506_20190818':
#         os.rename(file, file[:-4] + "_H1170")

#2017
for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/Hides 2017/1193/**/*.wav', recursive=True):
    # os.rename(file, file[:-16] + '.wav')
# #1193
    if file[62:75] == '1193_20170522':
        os.rename(file, file[:-4] + "_H3170.wav")

    if file[62:75] == '1193_20170523':
        os.rename(file, file[:-4] + "_H3170.wav")

    if file[62:75] == '1193_20170524':
        os.rename(file, file[:-4] + "_H3070.wav")

    if file[62:75] == '1193_20170525':
        os.rename(file, file[:-4] + "_H3070.wav")

    if file[62:75] == '1193_20170526':
        os.rename(file, file[:-4] + "_H3005.wav")

    if file[62:75] == '1193_20170527':
        os.rename(file, file[:-4] + "_H3005.wav")

# #1293
#     if file[62:75] == '1293_20170522':
#         os.rename(file, file[:-4] + "_H5005.wav")
#
#     if file[62:75] == '1293_20170523':
#         os.rename(file, file[:-4] + "_H5005.wav")
#
#     if file[62:75] == '1293_20170524':
#         os.rename(file, file[:-4] + "_H5070.wav")
#
#     if file[62:75] == '1293_20170525':
#         os.rename(file, file[:-4] + "_H5070.wav")
#
#     if file[62:75] == '1293_20170526':
#         os.rename(file, file[:-4] + "_H5170.wav")
#
#     if file[62:75] == '1293_20170527':
#         os.rename(file, file[:-4] + "_H5170.wav")
#
#     if file[62:75] == '1293_20170528':
#         os.rename(file, file[:-4] + "_H1005.wav")
#
#     if file[62:75] == '1293_20170529':
#         os.rename(file, file[:-4] + "_H1005.wav")
# #
# #1318
#
#     if file[62:75] == '1318_20170522':
#         os.rename(file, file[:-4] + "_H1170.wav")
#
#     if file[62:75] == '1318_20170523':
#         os.rename(file, file[:-4] + "_H1170.wav")
#
#     if file[62:75] == '1318_20170524':
#         os.rename(file, file[:-4] + "_H1070.wav")
#
#     if file[62:75] == '1318_20170525':
#         os.rename(file, file[:-4] + "_H1070.wav")
#
#     if file[62:75] == '1318_20170526':
#         os.rename(file, file[:-4] + "_H2070.wav")
#
#     if file[62:75] == '1318_20170527':
#         os.rename(file, file[:-4] + "_H2070.wav")
#
#     if file[62:75] == '1318_20170528':
#         os.rename(file, file[:-4] + "_H2170.wav")
#
#     if file[62:75] == '1318_20170529':
#         os.rename(file, file[:-4] + "_H2170.wav")

#1324
    #
    # if file[62:75] == '1324_20170522':
    #     os.rename(file, file[:-4] + "_H4005.wav")
    #
    # if file[62:75] == '1324_20170523':
    #     os.rename(file, file[:-4] + "_H4005.wav")
    #
    # if file[62:75] == '1324_20170524':
    #     os.rename(file, file[:-4] + "_H4070.wav")
    #
    # if file[62:75] == '1324_20170525':
    #     os.rename(file, file[:-4] + "_H4070.wav")

    if file[62:75] == '1324_20170526':
        os.rename(file, file[:-4] + "_H4170.wav")

    # if file[62:75] == '1324_20170527':
    #     os.rename(file, file[:-4] + "_H4170.wav")
    #
    # if file[62:75] == '1324_20170528':
    #     os.rename(file, file[:-4] + "_H6005.wav")
    #
    # if file[62:75] == '1324_20170529':
    #     os.rename(file, file[:-4] + "_H6005.wav")

# #1506
#     if file[62:75] == '1506_20170523':
#         os.rename(file, file[:-4] + "_H2005.wav")
#
#     if file[62:75] == '1506_20170524':
#         os.rename(file, file[:-4] + "_H2005.wav")
#
#     if file[62:75] == '1506_20170525':
#         os.rename(file, file[:-4] + "_H6170.wav")
#
#     if file[62:75] == '1506_20170526':
#         os.rename(file, file[:-4] + "_H6170.wav")
#
#     if file[62:75] == '1506_20170527':
#         os.rename(file, file[:-4] + "_H6070.wav")
#
#     if file[62:75] == '1506_20170528':
#         os.rename(file, file[:-4] + "_H6070.wav")

#2015
for file in glob.glob('/media/nottom/TOSHIBA EXT/joseph_dataset/Hides 2015/**/*.wav', recursive=True):
    print(file)
#1193
    # if file[62:75] == '1193_20150617':
    #     os.rename(file, file[:-4] + "_H2005.wav")

    # if file[62:75] == '1193_20150618':
    #     os.rename(file, file[:-4] + "_H2005.wav")

    # if file[62:75] == '1193_20150620':
    #     os.rename(file, file[:-4] + "_H5070.wav")
    #
    # if file[62:75] == '1193_20150621':
    #     os.rename(file, file[:-4] + "_H5070.wav")
    #
    # if file[62:75] == '1193_20150622':
    #     os.rename(file, file[:-4] + "_H4005.wav")
    #
    # if file[62:75] == '1193_20150623':
    #     os.rename(file, file[:-4] + "_H4005.wav")

# # 1293
    # if file[62:75] == '1293_20150616':
    #     os.rename(file, file[:-4] + "_H1070.wav")
    #
    # if file[62:75] == '1293_20150617':
    #     os.rename(file, file[:-4] + "_H2070.wav")
    #
    # if file[62:75] == '1293_20150618':
    #     os.rename(file, file[:-4] + "_H2070.wav")
    #
    # if file[62:75] == '1293_20150619':
    #     os.rename(file, file[:-4] + "_H6170.wav")
    #
    # if file[62:75] == '1293_20150620':
    #     os.rename(file, file[:-4] + "_H6170.wav")
    #
    # if file[62:75] == '1293_20150621':
    #     os.rename(file, file[:-4] + "_H4170.wav")
    #
    # if file[62:75] == '1293_20150622':
    #     os.rename(file, file[:-4] + "_H4170.wav")
    #
    # if file[62:75] == '1293_20150623':
    #     os.rename(file, file[:-4] + "_H1005.wav")

# #1318
#     if file[62:75] == '1318_20150616':
#         os.rename(file, file[:-4] + "_H1005.wav")
#
#     if file[62:75] == '1318_20150617':
#         os.rename(file, file[:-4] + "_H2170.wav")
#
#     if file[62:75] == '1318_20150618':
#         os.rename(file, file[:-4] + "_H2170.wav")
#
#     if file[62:75] == '1318_20150619':
#         os.rename(file, file[:-4] + "_H6005.wav")
#
#     if file[62:75] == '1318_20150620':
#         os.rename(file, file[:-4] + "_H6005.wav")
#
#     if file[62:75] == '1318_20150621':
#         os.rename(file, file[:-4] + "_H4070.wav")
#
#     if file[62:75] == '1318_20150622':
#         os.rename(file, file[:-4] + "_H4070.wav")
#
#     if file[62:75] == '1318_20150623':
#         os.rename(file, file[:-4] + "_H1070.wav")

# #1324
#     if file[62:75] == '1324_20150617':
#         os.rename(file, file[:-4] + "_H3005.wav")
#
#     if file[62:75] == '1324_20150618':
#         os.rename(file, file[:-4] + "_H3005.wav")
#
#     if file[62:75] == '1324_20150620':
#         os.rename(file, file[:-4] + "_H5005.wav")
#
#     if file[62:75] == '1324_20150621':
#         os.rename(file, file[:-4] + "_H5170.wav")
#
#     if file[62:75] == '1324_20150622':
#         os.rename(file, file[:-4] + "_H5170.wav")
#
#     if file[62:75] == '1324_20150623':
#         os.rename(file, file[:-4] + "_H1170.wav")
#
#     if file[62:75] == '1324_20150624':
#         os.rename(file, file[:-4] + "_H1170.wav")

#1506
    # if file[62:75] == '1506_20150617':
    #     os.rename(file, file[:-4] + "_H3070.wav")
    #
    # if file[62:75] == '1506_20150618':
    #     os.rename(file, file[:-4] + "_H3070.wav")
    #
    # if file[62:75] == '1506_20150619':
    #     os.rename(file, file[:-4] + "_H6070.wav")
    #
    # if file[62:75] == '1506_20150620':
    #     os.rename(file, file[:-4] + "_H6070.wav")
    #
    # if file[62:75] == '1506_20150621':
    #     os.rename(file, file[:-4] + "_H5005.wav")
    #
    # if file[62:75] == '1506_20150622':
    #     os.rename(file, file[:-4] + "_H3170.wav")
    #
    # if file[62:75] == '1506_20150623':
    #     os.rename(file, file[:-4] + "_H3170.wav")


#code to move all files into a single directory
for file in glob.glob('/media/nottom/TOSHIBA EXT/2021/Hides/**/*.wav', recursive=True):
    original = file
    destination = '' #<- to be decided!
    shutil.move(original, destination)
