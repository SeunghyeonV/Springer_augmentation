import time, os
import torch, torchvision
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


def train_acc(model, trainloader, device, train_batch, train_acc_list, total_train_acc):
    classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(100))
    class_total = list(0. for i in range(100))

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

    for i in range(len(train_acc_list)):
        for j in range(len(train_acc_list[i])):
            train_acc_list[i][j] = float(train_acc_list[i][j])

    print('Train accuracy of the network on the total trainset: {}/{} ({:.2f}%)'.format(correct, total,
                                                                                        100 * correct / total))  ## TOTAL ACCURACY
    total_train_acc.append(100 * correct / total)

def test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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
                                                                                      100 * correct / total))  ## TOTAL ACCURACY

def main():
    epochs = 200
    train_batch = 100
    test_batch = 100
    lr = 0.1
    clip_grad_norm = 0.1

    total_train_acc = []
    total_test_acc = []
    test_acc_list = [[] for i in range(100)]
    train_acc_list = [[] for i in range(100)]
    loss_list = []
    label_test_acc = [[] for i in range(100)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50().to(device)

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
    testset = torchvision.datasets.ImageFolder(root="C:/Users/seacl/Desktop/CIFAR100_reconstruction/cifar100/test",
                                                transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch, shuffle=True)

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        train(model, device, trainloader, trainset, optimizer, epoch, loss_list)
        t2 = time.time()
        train_acc(model, trainloader, device, train_batch, train_acc_list, total_train_acc)
        test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc)
        scheduler.step()

        epoch_time = t2 - t1
        print('Elapsed time per epoch: {:02d}h:{:02d}m:{:02d}s'.format(int(epoch_time // 3600),
                                                                       int(epoch_time % 3600 // 60),
                                                                       int(epoch_time % 60)))
    t3 = time.time()
    total_time_elapsed = t3 - t0
    print('Total elapsed time: {:02d}h:{:02d}m:{:02d}s'.format(int(total_time_elapsed // 3600),
                                                               int(total_time_elapsed % 3600 // 60),
                                                               int(total_time_elapsed % 60)))

    loss_list = [(float(round(i, 3))) for i in loss_list]
    for i in range(len(test_acc_list)):
        label_test_acc[i].append(test_acc_list[i])

    modelname = "resnet50"
    optimname = "SGD"
    parent_dir = "C:/Users/seacl/Desktop/CIFAR100_reconstruction/classification/pretrained_model"
    directory= "{}_{}_lr_{}/".format(modelname, optimname, lr)
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.mkdir(path)

    model_save = True
    if model_save:
        torch.save(model.state_dict(),
                   "{}/{}_{}_ep_{}_lr_{}_acc_{}%.pth".format(path, modelname, optimname,
                                                             epochs, lr, float(total_test_acc[-1])))

if __name__ == '__main__':
    main()

    
