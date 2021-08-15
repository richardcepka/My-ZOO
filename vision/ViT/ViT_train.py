import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from ViT_model import vit
from transformers import get_cosine_schedule_with_warmup

seed = 1297978
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

BATCH_SIZE = 256
LR = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIPPING = False
EPOCHS = 100
WARMUP_EPOCHS = 10
EVAL_EVERY = 10
CLASIFICATION_HEAD = "cls"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE = False


def acc(y, y_pred):
    return torch.sum(y_pred.argmax(1) == y).item()/len(y)


def load_cifar10() -> dict:
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    return {"trainloader": trainloader, "testloader": testloader, "num_classes": 10, "image_channels": 3}


def train(net, trainloader, testloader=None):
    """
    train: Adam, cosine schedule with linear warm up, gradient cliping = False
    """

    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=WARMUP_EPOCHS, num_training_steps=EPOCHS)

    for epoch in range(1, EPOCHS+1):
        train_loss = 0
        train_acc = 0
        for x, y in trainloader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            if GRADIENT_CLIPPING:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()

            train_acc += acc(y, y_pred)
            train_loss += loss.item()

        scheduler.step()

        print(
            f"epoch: {epoch}, train_loss: {format(train_loss/len(trainloader), '.4f')}, train_acc: {format(train_acc/len(trainloader), '.4f')}")

        if EVAL_EVERY > 0 and epoch % EVAL_EVERY == 0 and testloader != None or epoch == EPOCHS:
            evaluet(net, testloader)
            net.train()


def evaluet(net, testloader):
    net.eval()

    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    test_acc = 0
    for x, y in testloader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        y_pred = net(x)
        loss = criterion(y_pred, y)

        test_acc += acc(y, y_pred)
        test_loss += loss.item()

    print(
        f"test_loss: {format(test_loss/len(testloader), '.4f')}, test_acc: {format(test_acc/len(testloader), '.4f')}")


if __name__ == '__main__':
    loader = load_cifar10()
    net = vit(clasification_head=CLASIFICATION_HEAD).to(DEVICE)
    train(net=net,
          trainloader=loader["trainloader"],
          testloader=loader["testloader"])
    if SAVE:
        torch.save(net.state_dict(
        ), "pretrained_vit_cls.pth" if CLASIFICATION_HEAD == "cls" else "pretrained_vit_mean.pth")
