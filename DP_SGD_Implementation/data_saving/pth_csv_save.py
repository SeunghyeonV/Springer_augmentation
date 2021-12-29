import time, os
import torch, torchvision
from torchvision import transforms, datasets, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.models import VGG
from DPSGD_optimizer import DPSGD
import pandas as pd
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
                                                                                      100 * correct / total))  ## TOTAL ACCURACY


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
    model = torchvision.models.resnet18().to(device)
    # model = VGG('VGG11').to(device)
    optimizer = DPSGD(model.parameters(), l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, lr=lr,
                      batch_size=256)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.98)
    # optimizer = SGD(model.parameters(), lr=lr)

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.ImageFolder(root="/home/IIS/Desktop/CIFAR100_reconstruction/cifar100/train",
                                                transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True)
    testset = torchvision.datasets.ImageFolder(root="/home/IIS/Desktop/CIFAR100_reconstruction/cifar100/test",
                                               transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=train_batch, shuffle=True)


    # Save pretrained model and accuracy chart
    modelname = "resnet18"
    optimname = "DPSGD"
    parent_dir = "/home/IIS/Desktop/CIFAR100_reconstruction/DP/DP_result/"
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    pth_directory = "{}_{}_lr_{}_pth/".format(modelname, optimname, lr)
    csv_directory = "{}_{}_lr_{}_result/".format(modelname, optimname, lr)
    pth_path = os.path.join(parent_dir, pth_directory)
    csv_path = os.path.join(parent_dir, csv_directory)
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
        os.mkdir(csv_path)

    model_save = False
    result_save = True

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

        if model_save:
            torch.save(model.state_dict(),
                       "{}/{}_{}_ep_{}_lr_{}_n_{}_train_{}_test_{}%.pth".format(pth_path, modelname, optimname,
                                                                          epochs, lr, noise_multiplier,
                                                                          float(total_train_acc[-1]),
                                                                          float(total_test_acc[-1])))


        if result_save:
            loss_list = [(float(round(i, 3))) for i in loss_list]

            for i in range(len(test_acc_list)):
                label_test_acc[i].append(test_acc_list[i])

            # save the whole accuracy of each dataset to acc_savelist as an element
            acc_savelist = []

            # save the final epoch's accuracy of test_acc_list to testacc_last_elem
            testacc_last_elem = []
            for i in range(len(test_acc_list)):
                testacc_last_elem.append(float(test_acc_list[i][-1]))

            # save the whole accuracy list of each dataset to acc_savelist as nested form
            acc_savelist.append(testacc_last_elem)

            # must add one label to meet column size of df
            classes.append("Average")
            # Use pandas to save accuracy as organized form in csv
            df = pd.DataFrame(data=acc_savelist)
            df = df.T  # transpose it
            df.loc[len(df)] = total_test_acc[0]
            df.insert(loc=0, column="Label", value=classes)  # add label list as label to 0th column

            df_label_list = ['Label', 'Accuracy']
            df.columns = df_label_list  # define top rows

            # save result to csv file
            df.to_csv("{}/DP_model_acc_noise_01.csv".format(csv_path), encoding="utf_8_sig")

            print("Pth file save: ", model_save)
            print("accuracy data save: ", result_save)
            t3 = time.time()
            total_time_elapsed = t3 - t0
            print('Total elapsed time: {:02d}h:{:02d}m:{:02d}s'.format(int(total_time_elapsed // 3600),
                                                                       int(total_time_elapsed % 3600 // 60),
                                                                       int(total_time_elapsed % 60)))
            break



if __name__ == '__main__':
    main()
