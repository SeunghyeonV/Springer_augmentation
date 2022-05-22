import torch
from torchvision import datasets, transforms
import cv2
import os
from os import listdir
from os.path import isfile, join
toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()

def rotation():
    directory = ["cock", "electric_ray", "goldfish", "great_white_shark", "hammerhead", "hen", "ostrich", "stingray", "tench", "tiger_shark"]
    # Load file from address below
    trainset_address = "C:/Users/seacl/Desktop/recommendation_model/Imagenet/train/" #original trainset
    testset_address = "C:/Users/seacl/Desktop/recommendation_model/Imagenet/test/" #original testset
    train_directory = []
    test_directory = []
    train_file_list = [[] for _ in range(10)]
    test_file_list = [[] for _ in range(10)]
    trainpath_list = []
    testpath_list = []

    trainset_rotate_address = "C:/Users/seacl/Desktop/recommendation_model/rotated_Imagenet/train/" #augmented trainset
    testset_rotate_address = "C:/Users/seacl/Desktop/recommendation_model/rotated_Imagenet/test/" #autmented testset

    for i in range(len(directory)):
        path1 = os.path.join(trainset_rotate_address, directory[i])
        path2 = os.path.join(testset_rotate_address, directory[i])
        trainpath_list.append(path1)
        testpath_list.append(path2)

        if not os.path.exists(path1):
            os.makedirs(path1) #create 10 directories in testset_rotate_address

        if not os.path.exists(path2):
            os.makedirs(path2) #create 10 directories in testset_rotate_address



    for root, subdirectories, files in os.walk(trainset_address):
        for subdirectory in subdirectories:
            train_directory.append(os.path.join(root, subdirectory)) #load 10 labels from trainset_address

    for root, subdirectories, files in os.walk(testset_address):
        for subdirectory in subdirectories:
            test_directory.append(os.path.join(root, subdirectory)) #load 10 labels from testset_address

    for i in range(len(train_file_list)):
        for root, dirs, files in os.walk(train_directory[i]):
            for name in files:
                train_file_list[i].append(os.path.abspath(os.path.join(root, name))) #all file names from 10 subfolders

    for i in range(len(test_file_list)):
        for root, dirs, files in os.walk(test_directory[i]):
            for name in files:
                test_file_list[i].append(os.path.abspath(os.path.join(root, name))) #all file names from 10 subfolders


    for dir_idx in range(len(train_file_list)):
        for image_idx in range(len(train_file_list[dir_idx])):
            basename = os.path.basename(train_file_list[dir_idx][image_idx]) #xx.png
            image = cv2.imread(train_file_list[dir_idx][image_idx]) #read image
            (h, w) = image.shape[:2] #height, weight
            (cX, cY) = (w // 2, h // 2) #center
            M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0) #45 degree rotation
            rotated_image = cv2.warpAffine(image, M, (w, h)) 
            cv2.imwrite(os.path.join(trainpath_list[dir_idx], basename), rotated_image) #save augmented trainset

    for dir_idx in range(len(test_file_list)):
        for image_idx in range(len(test_file_list[dir_idx])):
            basename = os.path.basename(test_file_list[dir_idx][image_idx]) #xx.png
            image = cv2.imread(test_file_list[dir_idx][image_idx])
            (h, w) = image.shape[:2] #height, weight
            (cX, cY) = (w // 2, h // 2) #center
            M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0) #45 degree rotation
            rotated_image = cv2.warpAffine(image, M, (w, h))
            cv2.imwrite(os.path.join(testpath_list[dir_idx], basename), rotated_image)

def main():
    rotation()

if __name__ == '__main__':
    main()
