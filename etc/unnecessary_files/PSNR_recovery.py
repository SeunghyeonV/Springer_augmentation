#retrieves PSNR number from filename 

from os import listdir
from os.path import isfile, join
import os
import statistics as s

mypath = "/home/NewUsersDir/json4/json4/seunghyeon/final/recovered_dataset/randaug_M6/attack_accuracy/translateY/" # Directory of recovered image dataset

folder_list = []
for root, dirs, files in os.walk(mypath, topdown=False):
    for name in dirs:
        folder_list.append(os.path.join(mypath, name))
folder_list.sort()
print(folder_list[9])

file_list = []
mypath = folder_list[9]
filename = [f for f in listdir(mypath) if isfile(join(mypath, f))]
filename.sort()

label_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for j in range(len(label_list)):
    psnr = []
    mean_psnr = []
    for i in range(len(filename)):
        print(filename[i][-9:-4])
        psnr.append(filename[i][-9:-4])

    psnr = [float(i) for i in psnr]
    mean_psnr.append("{:.2f}".format(s.mean(psnr)))
    mean_psnr = [float(i) for i in mean_psnr]

    path = "/home/NewUsersDir/json4/json4/seunghyeon/final/recovered_dataset/randaug_M6/PSNR/translateY/" #Directory for saving PSNR files
    os.makedirs(path) if not os.path.exists(path) else None
    psnr_dir = "{}/psnr_list_{}.txt".format(path, label_list[j])
    mean_psnr_dir = "{}/mean_psnr_{}.txt".format(path, label_list[j])
    print(psnr, file=open(psnr_dir, "a"))
    print(mean_psnr, file=open(mean_psnr_dir, "a"))


