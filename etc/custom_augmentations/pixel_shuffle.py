from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import random


#Load file from address below
load_directory = 'E:/Image_folder/raw_cifar10_3000_images/truck'
imagename_list = [f for f in listdir(load_directory) if isfile(join(load_directory, f))]  # get filename 'XX.png'
filedir_list = [os.path.join(load_directory, imagename_list[i]) for i in
                range(len(imagename_list))]  # get full directory name + filename

def pixel_shuffle(image, imagename_list, idx):
    filename = imagename_list[idx]
    im = Image.open(image)  # Opens image
    imgwidth, imgheight = im.size  # image size
    box_list = []  # list for saving coordinates for puzzles
    height = imgheight // imgheight  # split image height
    width = imgwidth // imgwidth  # split image width

    for i in range(0, imgheight):
        for j in range(0, imgwidth):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            box_list.append(box)
    # print(box_list) #returns 1024 coordinates
    # print(len(box_list)) #returns 1024 as the image size is 32*32
    mixed_box_list = box_list.copy()
    for i in range(len(box_list)):
        if i % 4 == 0: #swap targeted pixels with adjacent pixels
            mixed_box_list[i], mixed_box_list[i+3] = mixed_box_list[i+3], mixed_box_list[i]


    empty_image = Image.new('RGB', (imgwidth, imgheight))  # Empty image to attach cropped images randomly
    for i in range(len(mixed_box_list)):
        puzzle = im.crop(box_list[i])
        empty_image.paste(puzzle, mixed_box_list[i])
        empty_image.save(filename)

    # plt.imshow(empty_image)
    # plt.show()


for i in range(len(filedir_list)):
    pixel_shuffle(filedir_list[i], imagename_list, i)
