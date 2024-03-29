#this file is for creating a csv out of the text files in a directory for training
import os
import csv
import glob

#this file will create a csv from all text files
import os
import csv
from pathlib import Path

#THIS IS THE OLD CODE FOR THE BINARY MODEL
folder = '/home/nottom/Documents/LinuxProject/multi_class_model/text_directories/training'
os.chdir('/home/nottom/Documents/LinuxProject/multi_class_model/text_directories/training')
with open('annotations_file_training.csv', 'w') as out_file:
    csv_out = csv.writer(out_file)
    # csv_out.writerow(['FileName', 'Content'])
    for file in Path('.').glob('*.txt'):
        join_path = os.path.join(folder, file)
        reader = open(join_path, 'r')
        content = reader.read()
        csv_out.writerow([str(file.with_suffix('.png')), open(str(file.absolute())).read().strip()])

#this is code for inference for binary model
folder = '/home/nottom/Documents/inference'
os.chdir('/home/nottom/Documents/inference')
with open('annotations_file_binaryinference.csv', 'w') as out_file:
    csv_out = csv.writer(out_file)
    # csv_out.writerow(['FileName', 'Content'])
    filelist = []
    for file in glob.glob('/home/nottom/Documents/inference/test/**/*.wav', recursive=True):
        print(file)
        filename = file[82:]
        print(filename)
        csv_out.writerow([filename])
print(filelist)

folder = '/home/nottom/Documents/inference/specgrams'
with open('annotations_file_binaryinference.csv', 'w') as out_file:
    csv_out = csv.writer(out_file)
    for file in os.listdir(folder):
        csv_out.writerow([file, 0])




import glob
import os
import csv
from pathlib import Path
#todo THIS IS THE CURRENT CODE FOR THE MULTILABEL MODEL
folder = '/home/nottom/Documents/LinuxProject/multi_class_model/text_directories/text_dir_training'
os.chdir('/home/nottom/Documents/LinuxProject/multi_class_model/text_directories/text_dir_training')
with open('annotations_file_training_multi.csv', 'w') as out_file:
    csv_out = csv.writer(out_file)
    # csv_out.writerow(['FileName', 'Content'])
    for file in Path('.').glob('*.txt'):
        join_path = os.path.join(folder, file)
        reader = open(join_path, 'r')
        content = reader.read()
        csv_out.writerow([str(file.with_suffix('.png')), content[0], content[3], content[6], content[9], content[12], content[15], content[18]])




# This code will transform all the one hot encoding values to a single integer
folder = '/home/nottom/Documents/LinuxProject/training_data_2009/all_textfiles'
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    reader = open(join_path, 'r')
    content = reader.read()
    # print(content)
    if file.endswith("1_.txt"):
         writer = open(join_path, 'w')
         writer.write("1")
    if file.endswith("0_.txt"):
         writer = open(join_path, 'w')
         writer.write("0")


# in case I need to switch values to binary
folder = '/home/nottom/Documents/LinuxProject/first_model/text_dir_test'
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

# This code will remove all segments that aren't of uniform size from spectrograms
folder = '/home/nottom/Documents/LinuxProject/first_model/valid_text' # make sure to do both training and validation text directories
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    if file.startswith('3591'):
        os.unlink(join_path)
    if file[0:4] == '3594':
        os.unlink(join_path)

#this code will remove all segments that aren't of uniform size from text_files
folder = '/home/nottom/Documents/LinuxProject/first_model/img_dir_valid' # make sure to do both training and validation image directories
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    if file.startswith('3591'):
        os.unlink(join_path)
    if file[0:4] == '3594':
        os.unlink(join_path)




#FOR REMOVING MULTI-LABELLED CHUNKS
import shutil
x=0
folder = '/home/nottom/Documents/Training/training_data/6/text' # make sure to do both training and validation text directories
for file in os.listdir(folder):
    join_path = os.path.join(folder, file)
    reader = open(join_path, 'r')
    content = reader.read()
    original = '/home/nottom/Documents/Training/training_data/6/text/' + file
    destination = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/6/' + file
    original_png = '/home/nottom/Documents/Training/training_data/6/specgrams/' + file[:-4]
    destination_png = '/home/nottom/Documents/Training/training_data/multi_labelled_examples/6/' + file[-4]
    if content != "0, 0, 0, 0, 0, 0, 1":
        shutil.move(original, destination)
        shutil.move(original_png, destination_png)











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
