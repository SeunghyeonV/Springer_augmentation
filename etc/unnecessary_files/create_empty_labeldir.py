import os

label_address = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test_sample"
label_list = os.listdir(label_address)
# print(label_list)

default_address = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test_50"

for i in range(len(label_list)):
    os.makedirs(os.path.join(default_address, label_list[i])) if not os.path.exists(os.path.join(default_address, label_list[i])) else None
        
