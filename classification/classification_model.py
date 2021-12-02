import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from zero.DP_optimizer import DPSGD

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
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

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
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

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

class Gaussian_noise(object):
    def __init__(self, mean=0, noise_multiplier=0, l2_norm_clip=0):
        self.mean = mean
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip

    def __call__(self, tensor):
        # return tensor + (torch.randn(tensor.size())*self.std) + self.mean
        return tensor + torch.zeros_like(tensor).normal_(mean=self.mean,
                                                         std=(self.noise_multiplier * self.l2_norm_clip))

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def main():
    epochs = 15
    train_batch = 50
    test_batch = 10
    lr = 0.5
    noise_multiplier = 1.0
    l2_norm_clip = 1.0

    total_train_acc = []
    total_test_acc = []
    test_acc_list = [[] for i in range(10)]
    train_acc_list = [[] for i in range(10)]
    loss_list = []
    label_test_acc = [[] for i in range(10)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18().to(device)
    optimizer = DPSGD(model.parameters(), l2_norm_clip=l2_norm_clip, noise_multiplier=noise_multiplier, lr=lr)

    # transform_train = transforms.Compose([transforms.ToTensor(),
    #                                       Gaussian_noise(mean=0., noise_multiplier=noise_multiplier,
    #                                                      l2_norm_clip=l2_norm_clip)])  # with noise

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=True)
    testset = torchvision.datasets.ImageFolder(
        root="C:/Users/seacl/Desktop/cifar10_3000_dataset",
        transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False)



    t0 = time.time()
    for epoch in range(epochs):
        t1 = time.time()
        train(model, device, trainloader, trainset, optimizer, epoch, loss_list)
        t2 = time.time()
        train_acc(model, device, trainloader, train_batch, train_acc_list, total_train_acc)
        test_acc(model, device, testloader, test_batch, test_acc_list, total_test_acc)

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
        label_test_acc[i].append(test_acc_list[i][-1])

    modelname = "resnet18"
    parent_dir = "F:/resnet50_DPSGD/"
    directory= "{}_lr_{}_nm_{}/".format(modelname, lr, noise_multiplier)
    path = os.path.join(parent_dir, directory)
    print(path)
    if not os.path.exists(path):
        os.mkdir(path)

    model_save = True
    if model_save:
        torch.save(model.state_dict(),
                   "{}/resnet50_ep_{}_lr_{}_acc_{}%.pth".format(
                       path, epochs, lr, float(total_train_acc[-1])))

    train_acc_list_dir = "{}/train_acc_list.txt".format(path)
    test_acc_list_dir = "{}/test_acc_list.txt".format(path)
    loss_list_dir = "{}/loss.txt".format(path)
    total_train_dir = "{}/total_train_acc.txt".format(path)
    total_test_dir = "{}/total_test_acc.txt".format(path)
    label_test_acc_dir = "{}/label_test_acc.txt".format(path)

    print(train_acc_list, file=open(train_acc_list_dir, "a"))
    print(test_acc_list, file=open(test_acc_list_dir, "a"))
    print(loss_list, file=open(loss_list_dir, "a"))
    print(total_train_acc, file=open(total_train_dir, "a"))
    print(total_test_acc, file=open(total_test_dir, "a"))
    print(label_test_acc, file=open(label_test_acc_dir, "a"))

if __name__ == '__main__':
    main()
