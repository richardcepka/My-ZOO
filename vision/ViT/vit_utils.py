import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from typing import Dict


def load_cifar10(batch_size: int = 128) -> dict:
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return {"trainloader": trainloader, "testloader": testloader, "num_classes": 10, "image_channels": 3}


def acc(y, y_pred):
    return torch.sum(y_pred.argmax(1) == y).item()/len(y)


def evaluate(net, testloader) -> Dict[str, int]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    test_acc = 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)

        y_pred = net(x)
        loss = criterion(y_pred, y)

        test_acc += acc(y, y_pred)

        test_loss += loss.item()
    return {"test_loss": test_loss/len(testloader), "test_acc": test_acc/len(testloader)}
