import os
import random
import shutil
from shutil import copyfile

# a function that selects 30 images from 100 images in each label folder of cifar-100
def data_sampling():
    
    # returns augmentations in list
    augmentation_list = os.listdir(
    "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100_output/randaug_M1/model_accuracy")
    
    # use test_copy to prevent the damage on original dataset
    testset_address = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test_copy"
    test_directory = []
    file_list = [[] for _ in range(100)]
    selected_files = []    
    
    
    # returns 100 label sources in list
    for root, subdirectories, files in os.walk(testset_address):
        for subdirectory in subdirectories:
            test_directory.append(os.path.join(root, subdirectory)) 
    # print(test_directory)

    # returns labels in list
    name_address = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test_copy"
    name_list = os.listdir(name_address)
    # print(name_list)

    # samples 30 file names from index_list 
    index_list = os.listdir("C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test/apple")
    sample_idx = random.sample(index_list, 50)
    # print(sample_idx)
    
    # returns full file path of sampled images
    l = [[] for _ in range(100)]
    for i in range(len(test_directory)):
        for j in range(len(sample_idx)):
            l[i].append(os.path.join(test_directory[i], sample_idx[j]))
    # print(l)
                   
    # creates 100 label destination folders in list
    destination_dir = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test_50"
    dest_dir = []
    for i in range(len(name_list)):
        if not os.path.exists(os.path.join(destination_dir, name_list[i])):
            os.mkdir(os.path.join(destination_dir, name_list[i]))
            dest_dir.append(os.path.join(destination_dir, name_list[i]))
    
    # copy sampled files to destination directories
    for i in range(len(test_directory)):
        for j in range(len(l[i])):
            # if copy2: copy files and paste it
            # shutil.copy2(l[i][j], dest_dir[i])
            
            # if move: move files from original directory to destination
            shutil.move(l[i][j], dest_dir[i])


def main():
    data_sampling()

if __name__ == '__main__':
    main()
