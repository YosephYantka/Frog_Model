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

