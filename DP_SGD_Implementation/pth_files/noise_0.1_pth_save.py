import time, os
import torch, torchvision
from torchvision import transforms, datasets, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.models import VGG
from DPSGD_optimizer import DPSGD
from torch.optim.lr_scheduler import StepLR

classes_dir = "/home/IIS/Desktop/CIFAR100_reconstruction/cifar100/train"
classes = os.listdir(classes_dir)

def train(model, device, trainloader, trainset, optimizer, epoch, loss_list):
    model.train()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    losses = []


    for batch_idx, batch in enumerate(trainloader):  # batch idx = 50000 / microbatches
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        # loss = F.cross_entropy(output, target)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 200 == 0:  # 1000 % 100 -> 5000, 10000, 15000, 20000, 25000 ...
            print('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {:.4f}'.format(epoch + 1, batch_idx * len(images),
                                                                                 len(trainloader.dataset),
                                                                                 100. * batch_idx / len(trainloader),
                                                                                 total_loss / (1.0 + batch_idx)))
    loss_list.append(float("{:.4f}".format(total_loss / (1.0 + batch_idx))))


def train_acc(model, device, trainloader, train_batch, train_acc_list, total_train_acc):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

    # model.load_state_dict(torch.load("C:/Users/Seunghyeon/Desktop/10.80.12.117/pretrained_models/Evaluation_purpose/resnet50_Cifar10_100ep_adam_acc_98.73%.pth"))
    # model = model.to(device)

    model.eval()
    train_loss = 0

    with torch.no_grad():
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            train_loss += F.cross_entropy(outputs, labels, reduction='sum').item()  # sum up batch loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()

            for i in range(train_batch):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    train_loss /= len(trainloader.dataset)
    for i in range(len(classes)):
        # print('Train accuracy of {}: {}/{} ({:.2f}%)'.format(classes[i], int(class_correct[i]), int(class_total[i]), 100 * class_correct[i] / class_total[i]))
        train_acc_list[i].append("{:.2f}".format(100 * class_correct[i] / class_total[i]))  ## EACH CLASS ACCURACY

    for i in range(len(train_acc_list)):
        for j in range(len(train_acc_list[i])):
            train_acc_list[i][j] = float(train_acc_list[i][j])

    print('Train set: Average loss: {:.4f}'.format(train_loss))
    total_train_acc.append(float("{:.2f}".format(100 * correct / total)))
    print('Train accuracy of the network on the total trainset: {}/{} ({:.2f}%)'.format(correct, total,
                                                                                        100 * correct / total))  ## TOTAL ACCURACY


def test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

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

    for i in range(len(test_acc_list)):
        for j in range(len(test_acc_list[i])):
            test_acc_list[i][j] = float(test_acc_list[i][j])


    print('Test set: Average loss: {:.4f}'.format(test_loss))
    total_test_acc.append(float("{:.2f}".format(100 * correct / total)))
    print('Test accuracy of the network on the total testset: {}/{} ({:.2f}%)'.format(correct, total,
                                                                                      100 * correct / total))

def main():
    epochs = 200
    train_batch = 100
    test_batch = 100
    lr = 0.2
    l2_norm_clip = 1.0
    noise_multiplier = 0.1

    total_train_acc = []
    total_test_acc = []
    test_acc_list = [[] for i in range(100)]
    train_acc_list = [[] for i in range(100)]
    loss_list = []
    label_test_acc = [[] for i in range(100)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = torchvision.models.resnet50().to(device)
    model = VGG('VGG11').to(device)
    optimizer = DPSGD(model.parameters(), l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, lr=lr,
                      batch_size=256)
    # optimizer = SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.98)

    transform = transforms.Compose([transforms.ToTensor()])
    # transform = transforms.Compose([transforms.RandomCrop(32, padding=4,padding_mode='reflect'),
    #                      transforms.RandomHorizontalFlip(),
    #                      transforms.ToTensor(),
    #                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #                     ])

    # transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    #                                       transforms.RandomHorizontalFlip(),
    #                                       transforms.RandomAffine((-20, 20)),
    #                                       transforms.RandomPerspective(distortion_scale=0.3, p=0.5, ),
    #                                       transforms.RandomVerticalFlip(),
    #                                       transforms.GaussianBlur(kernel_size=1, sigma=(0.1, 2.0)),
    #                                       transforms.ToTensor(),
    #                                       transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3),
    #                                                                value=0, inplace=False),
    #                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                                       ])

    trainset = torchvision.datasets.ImageFolder(root="/home/IIS/Desktop/CIFAR100_reconstruction/cifar100/train",
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True)
    testset = torchvision.datasets.ImageFolder(root="/home/IIS/Desktop/CIFAR100_reconstruction/cifar100/test",
                                               transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch, shuffle=True)

    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        train(model, device, trainloader, trainset, optimizer, epoch, loss_list)
        t2 = time.time()
        train_acc(model, device, trainloader, train_batch, train_acc_list, total_train_acc)
        test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc)
        scheduler.step()

        epoch_time = t2 - t1
        print('Elapsed time per epoch: {:02d}h:{:02d}m:{:02d}s'.format(int(epoch_time // 3600),
                                                                       int(epoch_time % 3600 // 60),
                                                                       int(epoch_time % 60)))

        if total_test_acc[-1] > 51.0:
            modelname = "VGG11"
            optimname = "DPSGD"
            parent_dir = "/home/IIS/Desktop/CIFAR100_reconstruction/DP/DP_result/pyvacy_pretrained_model"
            directory = "{}_{}_lr_{}/".format(modelname, optimname, lr)
            path = os.path.join(parent_dir, directory)
            if not os.path.exists(path):
                os.mkdir(path)

            model_save = True
            if model_save:
                torch.save(model.state_dict(),
                           "{}/{}_{}_ep_{}_lr_{}_n_{}_train_{}_test_{}%.pth".format(path, modelname, optimname,
                                                                              epochs, lr, noise_multiplier,
                                                                              float(total_train_acc[-1]),
                                                                              float(total_test_acc[-1])))
            break


if __name__ == '__main__':
    main()
