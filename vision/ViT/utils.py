import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
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

    trainset = torchvision.datasets.CIFAR10(root='./vision/data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./vision/data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return {"trainloader": trainloader, "testloader": testloader, "num_classes": 10, "image_channels": 3}


def acc(y, y_pred):
    return torch.sum(y_pred.argmax(1) == y).item()/len(y)


def evaluate(net, device, testloader) -> Dict[str, int]:

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


def train(net, config, device, trainloader, testloader=None):
    net.train()

    criterion = nn.CrossEntropyLoss()
    scheduler, optimizer = build_optimizer(config, net.parameters())

    for epoch in range(1, config["epochs"]+1):
        train_loss = 0
        train_acc = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            if config["gradient_clipping"]:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()

            train_acc += acc(y, y_pred)
            train_loss += loss.item()

        scheduler.step()

        print(
            f"epoch: {epoch}, train_loss: {format(train_loss/len(trainloader), '.4f')}, train_acc: {format(train_acc/len(trainloader), '.4f')}")

        if config["eval_every"] > 0 and epoch % config["eval_every"] == 0 and testloader != None or epoch == config["epochs"]:
            eval_metrics = evaluate(net, testloader)
            test_loss, test_acc = eval_metrics["test_loss"], eval_metrics["test_acc"]
            print(
                f"test_loss: {format(test_loss, '.4f')}, test_acc: {format(test_acc, '.4f')}")
            net.train()


def build_optimizer(config, params):

    if config['opt'] == 'adam':
        optimizer = optim.Adam(params,
                               lr=config['lr'],
                               weight_decay=config['weight_decay'])

    elif config['opt'] == 'sgd':
        optimizer = optim.SGD(params,
                              lr=config['lr'],
                              momentum=config['momentun'],
                              weight_decay=config['weight_decay'])

    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(params,
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    elif config['opt'] == 'adagrad':
        optimizer = optim.Adagrad(params,
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    if config['opt_scheduler'] == 'none':
        return None, optimizer

    elif config['opt_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=config['step_size'],
                                              gamma=config['gamma'])

    elif config['opt_scheduler'] == 'cos':
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config["warmup_epochs"],
                                                    num_training_steps=config["epochs"])
    return scheduler, optimizer
