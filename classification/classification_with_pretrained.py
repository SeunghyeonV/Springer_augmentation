import time, os
import torch, torchvision
from torchvision import transforms, datasets, models
import torch.nn.functional as F
import pandas as pd
from classification.resnet import resnet18
import sys
sys.path.append("C:/Users/seacl/Desktop/CIFAR100_reconstruction/classification/pretrained_model/")
from torchsummary import summary
import statistics as s

# The purpose of this code is to facilitate saving the test accuracy in csv file using pretrained model

classes_dir = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/train"
classes = os.listdir(classes_dir)

pretrained_model = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/classification/pretrained_model/resnet18-200ep_75.92%.pth"

def train_acc(model, trainloader, device, train_batch, train_acc_list, total_train_acc):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    train_loss = 0

    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            train_loss += F.cross_entropy(output, labels, reduction='sum').item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(train_batch):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    train_loss /= len(trainloader.dataset)
    for i in range(len(classes)):
        # print('Train accuracy of {}: {}/{} ({:.2f}%)'.format(classes[i], int(class_correct[i]), int(class_total[i]),
        #                                                      100 * class_correct[i] / class_total[i]))
        train_acc_list[i].append("{:.2f}".format(100 * class_correct[i] / class_total[i]))  ## EACH CLASS ACCURACY

    print('Train accuracy of the network on the total trainset: {}/{} ({:.2f}%)'.format(correct, total,
                                                                                        100 * correct / total))  ## TOTAL ACCURACY
    total_train_acc.append(100 * correct / total)

def test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    model.load_state_dict(torch.load(pretrained_model))
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += F.cross_entropy(outputs, labels, reduction='sum').item()  # sum up batch loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(test_batch):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    test_loss /= len(testloader.dataset)
    for i in range(len(classes)):
        # print('Test accuracy of {}: {}/{} ({:.2f}%)'.format(classes[i], int(class_correct[i]), int(class_total[i]), 100 * class_correct[i] / class_total[i]))
        test_acc_list[i].append("{:.2f}".format(100 * class_correct[i] / class_total[i]))  ## EACH CLASS ACCURACY

    print('Test set: Average loss: {:.4f}'.format(test_loss))
    total_test_acc.append(float("{:.2f}".format(100 * correct / total)))
    print('Test accuracy of the network on the total testset: {}/{} ({:.2f}%)'.format(correct, total,
                                                                                      100 * correct / total))  ## TOTAL ACCURACY

def main():
    train_batch = 100
    test_batch = 100

    total_train_acc = []
    total_test_acc = []
    test_acc_list = [[] for i in range(100)]
    train_acc_list = [[] for i in range(100)]

    # not used
    loss_list = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.resnet50().to(device)
    model = resnet18().to(device)
    # summary(model, (3,32,32))

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
                          transforms.RandomHorizontalFlip(),
                          transforms.RandomAffine((-20,20)),
                          transforms.RandomPerspective(distortion_scale=0.3, p=0.5,),
                          transforms.RandomVerticalFlip(),
                          transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0)),
                          transforms.ToTensor(),
                          transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])

    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])

    trainset = torchvision.datasets.ImageFolder(root="C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/train",
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True)

    testset_list = [
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M1/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M2/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M3/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M4/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M5/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M6/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M7/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M8/attack_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/autocontrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/brightness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/color",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/contrast",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/equalize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/invert",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/posterize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/rotate",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/sharpness",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/shearX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/shearY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/solarize",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/translateX",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/model_accuracy/translateY",
                    "E:/research/dataset/Image_folder/cifar100/randaugment_images/randaugment_M9/attack_accuracy/translateY",
                    ]

    # save the whole accuracy of each dataset to acc_savelist as an element
    acc_savelist = []
    avg_acc_savelist = []

    for t_idx in range(len(testset_list)):

        t1 = time.time()
        testset = torchvision.datasets.ImageFolder(root=testset_list[t_idx], transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch, shuffle=True)

        train_acc(model, trainloader, device, train_batch, train_acc_list, total_train_acc)
        test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc)

        # not used in this file
        loss_list = [(float(round(i, 3))) for i in loss_list]

        # save the final epoch's accuracy of test_acc_list to testacc_last_elem
        testacc_last_elem = []
        for i in range(len(test_acc_list)):
            testacc_last_elem.append(float(test_acc_list[i][-1]))

        # save the whole accuracy list of each dataset to acc_savelist as nested form
        acc_savelist.append(testacc_last_elem)
        avg_acc_savelist.append(total_test_acc[t_idx]) #return test accuracy of each testset

        t2 = time.time()
        single_image_elapsed = t2 - t1
        print('Elapsed time for testset_list[{}]: {:02d}h:{:02d}m:{:02d}s'.format(t_idx,
            int(single_image_elapsed // 3600), int(single_image_elapsed % 3600 // 60), int(single_image_elapsed % 60)))
            
    # must add one label to meet column size of df 
    classes.append("Average")
    
    # Use pandas to save accuracy as organized form in csv
    df = pd.DataFrame(data=acc_savelist)
    df = df.T  # transpose it
    df.loc[len(df)] = avg_acc_savelist
    df.insert(loc=0, column="Label", value=classes)  # add label list as label to 0th column

    df_label_list = ['Label']
    for i in range(len(testset_list)//2):
        df_label_list.append('MA')
        df_label_list.append('AA')
    df.columns = df_label_list  # define top rows   


    savefile = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/accuracy_table_m1m9.csv"  # filename
    df.to_csv(savefile, encoding="utf_8_sig")  # save file

if __name__ == '__main__':
    main()
