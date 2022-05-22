import torch
import matplotlib.pyplot as plt
import socket
from utils.models import VGG
import time, datetime
from torchvision.utils import save_image
from torchvision import transforms
import os
from os import listdir
from os.path import isfile, join
from utils.randaugment import *
import inversefed
from utils.custom_dataset import CustomImageDataset
import statistics as s
######################################################################################################
## Global params
config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=4000,
              total_variation=1e-6,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print('Currently evaluating -------------------------------:')
print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
print(f'GPU : {torch.cuda.get_device_name(device=device)}')

defs = inversefed.training_strategy('conservative')
loss_fn, trainloader, validloader = inversefed.construct_dataloaders('CIFAR10', defs)
dm = torch.as_tensor(inversefed.consts.cifar10_mean)[:, None, None].to(device)
ds = torch.as_tensor(inversefed.consts.cifar10_std)[:, None, None].to(device)
toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()

model = VGG('VGG11').to(device)  # Fast and clear recovery available - takes only 10 seconds
# pretrained_model = "/home/IIS/Desktop/seunghyeon/test/0417/VGG11_ep_200_lr_0.001_acc_69.61%.pth"
# model.load_state_dict(torch.load(pretrained_model))
model.eval();

augmentation_list = ['autocontrast', 'brightness', 'color', 'contrast', 'equalize',
                     'invert', 'posterize', 'rotate', 'sharpness', 'shearX', 'shearY',
                     'solarize', 'translateX', 'translateY']

label_address = "/home/iis/Desktop/CIFAR100_reconstruction/cifar100/test_10/"
label_list = os.listdir(label_address)
label_list.sort()
# print(label_list) # -> returns cifar-100 labels


transforms_list = [transforms.Compose([CIFAR10Policy_autocontrast(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_brightness(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_color(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_contrast(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_equalize(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_invert(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_posterize(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_rotate(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_sharpness(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_shearX(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_shearY(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_solarize(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_translateX(magnitude=2), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_translateY(magnitude=2), transforms.ToTensor()]),
                   ]


######################################################################################################
def file_loader():
    ### Load files and directory addresses
    ## get filename

    input_address_first = []
    for i in range(len(label_list)):
        input_address_first.append(os.path.join(label_address, label_list[i]))
    # print(input_address_first) # -> returns full address of source directories

    # load file names in list for saving reconstructed files
    filenames = [f for f in listdir(input_address_first[0]) if isfile(join(input_address_first[0], f))]
    # print(filenames) # -> returns filename i.e. 0011.png

    filename_list = []
    for i in range(len(input_address_first)):
        for j in range(len(filenames)):
            filename_list.append(os.path.join(input_address_first[i], filenames[j]))
    # print(filename_list) # -> returns full path of images
    # print(len(filename_list))
    filename_list.sort()
    return filename_list


#################################################################################################
def input_savedir_M2():
    ## get filename
    default_address = "/home/iis/Desktop/CIFAR100_reconstruction/cifar100_output/randaug_M2/model_accuracy/"
    os.makedirs(default_address) if not os.path.exists(default_address) else None
    input_address_first = []

    # generates C:\Users\seacl\Desktop\CIFAR100_reconstruction\cifar100_output\randaug_M2\model_accuracy\ 14 aug...
    for i in range(len(augmentation_list)):
        input_address_first.append(os.path.join(default_address, augmentation_list[i]))
        os.makedirs(input_address_first[i]) if not os.path.exists(input_address_first[i]) else None
    # # print(input_address_first) # returns full address of source directories

    input_address_final = [[] for _ in range(len(augmentation_list))]
    for i in range(len(input_address_first)):
        for j in range(len(label_list)):
            input_address_final[i].append(os.path.join(input_address_first[i], label_list[j]))
            os.makedirs(input_address_final[i][j]) if not os.path.exists(input_address_final[i][j]) else None

    input_address_final.sort()
    return input_address_final

def output_savedir_M2():
    ## get filename
    default_address = "/home/iis/Desktop/CIFAR100_reconstruction/cifar100_output/randaug_M2/attack_accuracy/"
    os.makedirs(default_address) if not os.path.exists(default_address) else None
    output_address_first = []

    # generates C:\Users\seacl\Desktop\CIFAR100_reconstruction\cifar100_output\randaug_M2\model_accuracy\ 14 aug...
    for i in range(len(augmentation_list)):
        output_address_first.append(os.path.join(default_address, augmentation_list[i]))
        os.makedirs(output_address_first[i]) if not os.path.exists(output_address_first[i]) else None
    # # print(input_address_first) # returns full address of source directories

    output_address_final = [[] for _ in range(len(augmentation_list))]
    for i in range(len(output_address_first)):
        for j in range(len(label_list)):
            output_address_final[i].append(os.path.join(output_address_first[i], label_list[j]))
            os.makedirs(output_address_final[i][j]) if not os.path.exists(output_address_final[i][j]) else None

    output_address_final.sort()
    return output_address_final

def psnr_savedir_M2():
    # psnr files are saved in augmentation folder without label subdirectories for convenience
    default_address = "/home/iis/Desktop/CIFAR100_reconstruction/cifar100_output/randaug_M2/psnr"
    os.makedirs(default_address) if not os.path.exists(default_address) else None

    psnr_address_final = []
    for i in range(len(augmentation_list)):
        full_path = os.path.join(default_address, augmentation_list[i])
        os.makedirs(full_path) if not os.path.exists(full_path) else None
        psnr_address_final.append(full_path)

    psnr_address_final.sort()
    return psnr_address_final

#################################################################################################
def input_plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        save_image(tensor, ("{}/{}".format(input_savepath[t][f//num_files], f_loader[f][-8:])))
        # plt.imshow(tensor[0].permute(1, 2, 0).cpu())
        # plt.show()
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())

def output_plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        save_image(tensor, ("{}/{}".format(output_savepath[t][f//num_files], f_loader[f][-8:])))
        # plt.imshow(tensor[0].permute(1, 2, 0).cpu())
        # plt.show()
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())

#################################################################################################


f_loader = file_loader()
input_savepath = input_savedir_M2()
output_savepath = output_savedir_M2()
psnr_savepath = psnr_savedir_M2()
num_files = 10 # change the variable depending on the number of images per label

for t in range(len(transforms_list)): # 14
    psnr_list = [[] for _ in range(100)] # reset when every new augmentation starts
    mean_psnr_list = [[] for _ in range(100)]

    for f in range(len(f_loader)): # 3000
        print("currently reconstructing: ")
        print("image index: ", f) # if disconnected check this number and start from here
        print("transform: ", augmentation_list[t])
        print("label: ", label_list[f//num_files])
        print("filename: ", f_loader[f][-8:])

        t1 = time.time()
        ground_truth_image = transforms_list[t](Image.open(f_loader[f])).to(device, dtype=torch.float)
        ground_truth = ground_truth_image.sub(dm).div(ds).unsqueeze(0).contiguous()
        labels = torch.as_tensor((1,)).to(device)
        input_plot(ground_truth)

        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]

        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))
        output_plot(output)

        test_mse = (output.detach() - ground_truth).pow(2).mean()
        feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()
        test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)
        psnr_list[f//num_files].append(test_psnr)

        t2 = time.time()
        single_image_elapsed = t2 - t1
        print('Elapsed time for image {}: {:02d}h:{:02d}m:{:02d}s'.format(f,
            int(single_image_elapsed // 3600), int(single_image_elapsed % 3600 // 60), int(single_image_elapsed % 60)))

    # psnr_list and mean_psnr_list have nested form, 100 inner lists each
    # mean_psnr_list[i] saves the mean of psnr_list[i]
    #
    psnr_filename_list = []
    for l in range(len(label_list)):
        mean_psnr_list[l].append(round(s.mean(psnr_list[l]), 2))
        psnr_filename = os.path.join(psnr_savepath[t], "mean_psnr_{}.txt").format(label_list[l])
        psnr_filename_list.append(psnr_filename)
        print(mean_psnr_list[l][0], file=open(psnr_filename_list[l], "a"))



# # # ### PSNR_saving pseudocode
# for t in range(1):
#     psnr_list = [[] for _ in range(100)]
#     mean_psnr_list = [[] for _ in range(100)]
#
#     for f in range(3000):  # 3000
#         test_psnr = round(random.uniform(1, 2), 2)
#         psnr_list[f//num_files].append(test_psnr)
#     # print(psnr_list)
#
#     psnr_filename_list = []
#     for l in range(len(label_list)):
#         mean_psnr_list[l].append(round(s.mean(psnr_list[l]), 2)) # compute mean of psnr_list
#         psnr_filename = os.path.join(psnr_savepath[t], "mean_psnr_{}.txt").format(label_list[l]) # create filename for saving
#         psnr_filename_list.append(psnr_filename) # save filenames in list
#         print(mean_psnr_list[l][0], file=open(psnr_filename_list[l], "a")) # create psnr files

