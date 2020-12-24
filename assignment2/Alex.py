import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset
# import pickle
# import os
# import copy
# from sklearn.manifold import TSNE

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.models = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            # nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            # nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.models(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


def find_model(net, trainset, trainloader, testloader, criterion, optimizer, device, epochsize=10, batch_size=500):
    trainloss_epoch = []
    testloss_epoch = []
    trainacc_epoch = []
    testacc_epoch = []

    trainloss_batch = []
    for epoch in range(epochsize):
        running_loss = 0.
        for i, data in enumerate(
                torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            trainloss_batch.append(loss.item())

            loss.backward()
            optimizer.step()
        #             print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))

        with torch.no_grad():

            trainloss = 0.0
            traincorrect = 0
            num = 0

            for inputs, classes in trainloader:
                inputs, classes = inputs.to(device), classes.to(device)
                y = net(inputs)
                # loss = criterion(y, classes)
                _, predicted = torch.max(y.data, 1)

                num += classes.size(0)
                # trainloss += loss.item()
                traincorrect += (predicted == classes).sum().item()

            # trainloss_epoch.append(trainloss / num)
            trainacc_epoch.append(traincorrect / num)
            print(trainacc_epoch)

            testloss = 0.0
            testcorrect = 0
            num = 0

            for inputs, classes in testloader:
                inputs, classes = inputs.to(device), classes.to(device)

                y = net(inputs)
                # loss = criterion(y, classes)
                _, predicted = torch.max(y.data, 1)

                num += classes.size(0)
                # testloss += loss.item()
                testcorrect += (predicted == classes).sum().item()

            # testloss_epoch.append(testloss / num)
            testacc_epoch.append(testcorrect / num)
            print(testacc_epoch)

    print('Finished Training')

    return net, trainacc_epoch, testacc_epoch

    # trainloss_epoch = np.array(trainloss_epoch)
    # testloss_epoch = np.array(testloss_epoch)
    # trainacc_epoch = np.array(trainacc_epoch)
    # testacc_epoch = np.array(testacc_epoch)
    # trainloss_batch = np.array(trainloss_batch)
    # np.save('./result/norm_trainloss_epoch',trainloss_epoch )
    # np.save('./result/norm_testloss_epoch',testloss_epoch)
    # np.save('./result/sgd_trainacc_epoch',trainacc_epoch)
    # np.save('./result/sgd_testacc_epoch',testacc_epoch)
    # np.save('./result/sgd_trainloss_batch',trainloss_batch)
    # torch.save(net, './result/sgd_cifar10.pkl')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


lrlist = [10,500]
totallr = []
for i in lrlist:
    net = AlexNet()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net, trainacc_epoch, testacc_epoch  = find_model(net,trainset,trainloader,testloader,criterion,optimizer,device,epochsize=10,batch_size = i)
    totallr.append(np.array(trainacc_epoch))
    totallr.append(np.array(testacc_epoch))

totallr = np.array(totallr)
np.save('./result/totalbatch',totallr)