#this file is for creating a csv out of the text files in a directory for training
import os
import csv

# This code will transform all the one hot encoding values to a single integer



folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_training'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    reader = open(join_path, 'r')
    content = reader.read()
    print(content)
    if content == '1':
         writer = open(join_path, 'w')
         writer.write("0")
      if content == '2':
         writer = open(join_path, 'w')
         writer.write("1")

# This code will remove all segments that aren't of uniform size
folder = '/home/nottom/Documents/LinuxProject/first_model/valid_text' # make sure to do both training and validation text directories
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    if file.startswith('3591'):
        os.unlink(join_path)
    if file[0:4] == '3594':
        os.unlink(join_path)

folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_valid' # make sure to do both training and validation image directories
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    if file.startswith('3591'):
        os.unlink(join_path)
    if file[0:4] == '3594':
        os.unlink(join_path)

#this file will create a csv from all text files
import os
import csv
from pathlib import Path
folder = '/home/nottom/Documents/LinuxProject/first_model/text_dir_valid'
os.chdir('/home/nottom/Documents/LinuxProject/first_model/text_dir_valid')
with open('annotations_file_valid.csv', 'w') as out_file:
    csv_out = csv.writer(out_file)
    # csv_out.writerow(['FileName', 'Content'])
    for fileName in Path('.').glob('*.txt'):
        # lala = fileName
        # csv_out.writerow([str(fileName) + ',png',open(str(fileName.absolute())).read().strip()])
        csv_out.writerow([str(fileName.with_suffix('.png')), open(str(fileName.absolute())).read().strip()])




















#
# #write each text file into a csv
# for file in os.listdir(text_files):
#     join_path = os.path.join(text_files, file)
#     f = open(join_path, 'r')
#     content = f.read()
#     filename = str(file[0:-4])
#
#     with open('/home/nottom/Documents/LinuxProject/training_data/text/hopeful.csv', "w") as csv_file: #<destination for the csv
#         writer = csv.writer(csv_file, delimiter=',')
#         for line in list:
#             writer.writerow(content)
#
#
# #^this doesn't really work but maybe it will one day
#
# import glob
# os.chdir('/home/nottom/Documents/LinuxProject/training_data/text/background_2019_2021')
# glob.glob("*.txt")
# globlet = glob.glob("*.txt")
# print(globlet)
#
#
# csv_list = []
# for file in os.listdir(text_files):
#     join_path = os.path.join(text_files, file)
#     f = open(join_path, 'r')
#     content = f.read()
#     splitlist = content.split(', ')
#     splitlist = [int(x) for x in splitlist]
#     csv_list.append(splitlist)
# #
# print(csv_list)
# type(csv_list[2][2])
# #
# # def split(list_a, chunk_size):
# #     for i in range(0, len(list_a), chunk_size):
# #         yield list_a[i:i + chunk_size]
# #
# # chunk_size = 1
# # csv_list_oflists = list(split(csv_list, chunk_size))
# # print(csv_list_oflists)
#
# with open("/home/nottom/Documents/LinuxProject/training_data/text/hopeful.csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerows(csv_list)
#
# #new one
#
#
