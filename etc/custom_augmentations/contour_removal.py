import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def contour_removal():
    directory = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    # Load file from address below
    load_address = "C:/Users/seacl/Desktop/raw_cifar10_3000_images/"
    save_address = "C:/Users/seacl/Desktop/recommendation_model/augmented_dataset/contour_removal/"
    load_directory = []
    save_directory = []

    for root, subdirectories, files in os.walk(load_address):
        for subdirectory in subdirectories:
            load_directory.append(os.path.join(root, subdirectory)) #load 10 labels from load_address

    for i in range(len(directory)):
        path = os.path.join(save_address, directory[i])
        os.makedirs(path) #create 10 directories in save_address

    for root, subdirectories, files in os.walk(save_address):
        for subdirectory in subdirectories:
            save_directory.append(os.path.join(root, subdirectory)) #load 10 labels from save_address

    for dir_idx in range(len(load_directory)):
        for image_idx in range(len(os.listdir(load_directory[dir_idx]))):
            imagename_list = [f for f in listdir(load_directory[dir_idx]) if
                              isfile(join(load_directory[dir_idx], f))]  # get filename 'XX.png'

            filedir_list = [os.path.join(load_directory[dir_idx], imagename_list[image_idx]) for image_idx in
                            range(len(imagename_list))]  # get full directory name + filename

            img = cv2.imread(filedir_list[image_idx])
            imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, threshold = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
            # img_binary = cv2.bitwise_not(threshold)

            mask = np.zeros(img.shape, np.uint8)  # black image for masking
            contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            visible_contour = cv2.drawContours(img, contours, -1, (0, 0, 0), 1)  # visible contour on real image
            # cv2.imshow("visible_contour", visible_contour)
            # cv2.waitKey(0)

            filename = imagename_list[image_idx]
            cv2.imwrite(os.path.join(save_directory[dir_idx], filename), img)

def main():
    contour_removal()

if __name__ == '__main__':
    main()
