# use this method to create a directory 
# directory include 14 augmentations and 100 CIFAR-100 label subfolders in each augmentation folder, total 1400 nested directories
# List of 1400 directories are provided

def make_output_dir():
    ## get filename
    name_address = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test"
    name_list = os.listdir(name_address)
    # print(name_list) # -> filename ['apple','aquarium_fish',...]

    ## get augmentation names
    target_address = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100_output/test"
    target_address_innerdir = os.listdir(target_address)
    # print(target_address_innerdir) #-> ['autocontrast', 'brightness', ...]

    ## get augmentation directory address
    ## ['C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100_output/test//autocontrast', ... ]
    full_savepath = []
    for i in range(len(target_address_innerdir)):
        full_savepath.append(os.path.join(target_address, target_address_innerdir[i]))
    # print(full_savepath)

    ## make augmentation directories with subdirectories of name_list
    ## "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100_output/test/autocontrast/apple"
    print("Start making directories: ")
    for dir in full_savepath:
        for filename in name_list:
            if not os.path.exists(os.path.join(dir,filename)):
                os.mkdir(os.path.join(dir, filename))
            else:
                break
    print("Done!")

    ## return the list of 1400 directories
    ## ['C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100_output/test\\autocontrast\\apple', ...]
    full_dir = []
    for i in range(len(full_savepath)):
        for root, subdirectories, files in os.walk(full_savepath[i]):
            for subdirectory in subdirectories:
                full_dir.append(os.path.join(root, subdirectory))
    # print(full_dir)
    # print(len(full_dir))
