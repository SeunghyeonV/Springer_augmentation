import torch
import matplotlib.pyplot as plt
import socket
from utils.models import VGG
import inversefed
import time, datetime
from torchvision.utils import save_image
from torchvision import transforms
import os
from os import listdir
from os.path import isfile, join
from utils.randaugment import *


#################################################################################################
def file_loader():
    ### Load files and directory addresses
    ## get filename
    label_address = "/home/json/Desktop/CIFAR100_reconstruction/cifar100/sampled_test/"
    label_list = os.listdir(label_address)
    # print(label_list) # -> returns cifar-100 labels

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
def input_savedir_M1():
    ## get filename
    default_address = "/home/json/Desktop/CIFAR100_reconstruction/cifar100_output/randaug_M7/model_accuracy/"
    augmentation_list = os.listdir(default_address)
    augmentation_list.sort()
    # print(augmentation_list) # -> returns augmentation list

    input_address_first = []
    for i in range(len(augmentation_list)):
        input_address_first.append(os.path.join(default_address, augmentation_list[i]))
    input_address_first.sort()
    # print(input_address_first) # returns full address of source directories

    labels = os.listdir(input_address_first[0])
    labels.sort()
    # print(labels) # -> returns labels

    input_address_final = [[] for _ in range(len(augmentation_list))]

    for i in range(len(input_address_first)):
        for j in range(len(labels)):
            input_address_final[i].append(os.path.join(input_address_first[i], labels[j]))
    # print(input_address_final)
    # print(len(input_address_final))
    input_address_final.sort()
    return input_address_final


#################################################################################################
def output_savedir_M1():
    ## get filename
    default_address = "/home/json/Desktop/CIFAR100_reconstruction/cifar100_output/randaug_M7/attack_accuracy/"
    augmentation_list = os.listdir(default_address)
    # print(augmentation_list) # -> returns augmentation list
    augmentation_list.sort()

    output_address_first = []
    for i in range(len(augmentation_list)):
        output_address_first.append(os.path.join(default_address, augmentation_list[i]))
    output_address_first.sort()
    # print(output_address_first) # returns full address of source directories

    labels = os.listdir(output_address_first[0])
    labels.sort()
    # print(labels) # -> returns labels

    output_address_final = [[] for _ in range(len(augmentation_list))]

    for i in range(len(output_address_first)):
        for j in range(len(labels)):
            output_address_final[i].append(os.path.join(output_address_first[i], labels[j]))
    # print(output_address_final)
    # print(len(output_address_final))
    output_address_final.sort()
    return output_address_final


#################################################################################################
def return_augmentations():
    # augmentation_list = ['autocontrast', 'brightness', 'color', 'contrast', 'equalize',
    #                      'invert', 'posterize', 'rotate', 'sharpness', 'shearX', 'shearY',
    #                      'solarize', 'translateX', 'translateY']
    augmentation_list = ['brightness', 'color', 'contrast',
                         'posterize', 'rotate', 'sharpness', 'shearX', 'shearY',
                         'solarize', 'translateX', 'translateY']
    return augmentation_list


def return_labels():
    label_address = "/home/json/Desktop/CIFAR100_reconstruction/cifar100/sampled_test/"
    label_list = os.listdir(label_address)
    label_list.sort()
    return label_list


def return_filenames():
    f_dir = "/home/json/Desktop/CIFAR100_reconstruction/cifar100/sampled_test/apple/"
    filenames = [f for f in listdir(f_dir) if isfile(join(f_dir, f))]
    filenames.sort()
    return filenames


#################################################################################################
def input_plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        save_image(tensor, ("{}/{}".format(input_savepath[t][l], filenames[f])))
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0] * 12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
            save_image(tensor, ("{}/{}".format(input_savepath[t][l], filenames[f])))


def output_plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        save_image(tensor, ("{}/{}".format(output_savepath[t][l], filenames[f])))
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0] * 12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
            save_image(tensor, ("{}/{}".format(output_savepath[t][l], filenames[f])))


#################################################################################################

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

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
print('Currently evaluating -------------------------------:')
print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
print(f'GPU : {torch.cuda.get_device_name(device=device)}')

toTensor = transforms.ToTensor()
toPIL = transforms.ToPILImage()
defs = inversefed.training_strategy('conservative')
loss_fn, trainloader, validloader = inversefed.construct_dataloaders('CIFAR10', defs)
dm = torch.as_tensor(inversefed.consts.cifar10_mean)[:, None, None].to(device)
ds = torch.as_tensor(inversefed.consts.cifar10_std)[:, None, None].to(device)

model = VGG('VGG11').to(device)  # Fast and clear recovery available - takes only 10 seconds
# activate the line below to use pretrained model
# model.load_state_dict(torch.load('C:/Users/seacl/Desktop/test/dp_result/DPSGD_sigma_0.5_acc_47.256%.pth'))
model.eval();

input_savepath = input_savedir_M1()
output_savepath = output_savedir_M1()
f_loader = file_loader()
label_list = return_labels()
filenames = return_filenames()
augmentations = return_augmentations()

transforms_list = [#transforms.Compose([CIFAR10Policy_autocontrast(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_brightness(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_color(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_contrast(magnitude=7), transforms.ToTensor()]),
                   #transforms.Compose([CIFAR10Policy_equalize(magnitude=7), transforms.ToTensor()]),
                   #transforms.Compose([CIFAR10Policy_invert(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_posterize(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_rotate(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_sharpness(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_shearX(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_shearY(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_solarize(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_translateX(magnitude=7), transforms.ToTensor()]),
                   transforms.Compose([CIFAR10Policy_translateY(magnitude=7), transforms.ToTensor()]),
                   ]


for t in range(len(transforms_list)):
    for l in range(len(label_list)):
        for f in range(len(filenames)):
            print("transform: ", augmentations[t])
            print("label: ", label_list[l])
            print("filename: ", f_loader[f][-8:])
            t1 = time.time()
            ground_truth_image = transforms_list[t](Image.open(f_loader[f])).to(device, dtype=torch.float)
            ground_truth = ground_truth_image.sub(dm).div(ds).unsqueeze(0).contiguous()
            labels = torch.as_tensor((1,)).to(device)
            input_plot(ground_truth)

            local_lr = 1e-4
            local_steps = 5
            use_updates = True

            model.zero_grad()
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth, labels,
                                                                               lr=local_lr, local_steps=local_steps,
                                                                               use_updates=use_updates)
            input_parameters = [p.detach() for p in input_parameters]

            rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_steps, local_lr, config,
                                                         use_updates=use_updates)
            output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=(3, 32, 32))
            output_plot(output)

            test_mse = (output.detach() - ground_truth).pow(2).mean()
            feat_mse = (model(output.detach()) - model(ground_truth)).pow(2).mean()
            test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)

            t2 = time.time()
            print(
                "Elapsed time for {}_{}_{}: {:.2f}s".format(augmentations[t], label_list[l], f_loader[f][-8:], t2 - t1))
#
#
