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

def puzzle_shuffle(image, ver_split, hor_split, imagename_list, idx):
    filename = imagename_list[idx]
    im = Image.open(image)  # Opens image
    imgwidth, imgheight = im.size  # image size
    empty_image = Image.new('RGB', (imgwidth, imgheight))  # Empty image to attach cropped images randomly
    box_list = []  # list for saving coordinates for puzzles
    height = imgheight // ver_split  # split image height
    width = imgwidth // hor_split  # split image width
    same_split = False

    for i in range(0, ver_split):
        for j in range(0, hor_split):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            box_list.append(box)

    randbox = random.sample(box_list, len(box_list))  # shuffle the sequence of cropped image
    while randbox == box_list:
        randbox = random.sample(box_list, len(box_list))  # doing this prevents allocating the block at the original spot

    for i in range(len(box_list)):
        puzzle = im.crop(box_list[i])
        # augmented_image = Image.Image.paste(empty_image, puzzle,
        #                   randbox[i])  # attaches puzzles on empty image. now need to find a way to randomly attach them
        empty_image.paste(puzzle, randbox[i])
        empty_image.save(filename)

    # plt.imshow(empty_image)
    # plt.show()
#
for i in range(len(filedir_list)):
    split_shuffle(filedir_list[i], 2, imagename_list, i)
    split_shuffle(filedir_list[i], 2, 1, imagename_list, i) # if the number of vertical and horizontal split is different:
