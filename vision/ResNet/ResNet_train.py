import torch
import torch.optim as optim
import torch.nn as nn
from ResNet_model import resnet18
from resnet_utils import acc, evaluate, load_cifar10
from resnet_config import config

seed = 1297978
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, trainloader, testloader=None):
    """
    train: SGD, linear scheduler
    """

    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["train_config"]["lr"],
                          momentum=0.9, weight_decay=config["train_config"]["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=config["train_config"]["epochs"]//3, gamma=0.1)

    for epoch in range(1, config["train_config"]["epochs"]+1):
        train_loss = 0
        train_acc = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_acc += acc(y, y_pred)
            train_loss += loss.item()

        scheduler.step()

        print(
            f"epoch: {epoch}, train_loss: {format(train_loss/len(trainloader), '.4f')}, train_acc: {format(train_acc/len(trainloader), '.4f')}")

        if config["train_config"]["eval_every"] > 0 and epoch % config["train_config"]["eval_every"] == 0 and testloader != None or epoch == config["train_config"]["epochs"]:
            eval_metrics = evaluate(net, testloader)
            test_loss, test_acc = eval_metrics["test_loss"], eval_metrics["test_acc"]
            print(
                f"test_loss: {format(test_loss, '.4f')}, test_acc: {format(test_acc, '.4f')}")
            net.train()


if __name__ == '__main__':
    loader = load_cifar10(batch_size=config["train_config"]["bath_size"])
    net = resnet18(image_channels=config["model_config"]["image_channels"],
                   num_classes=config["model_config"]["num_classes"],
                   bottleneck=config["model_config"]["bottleneck"]).to(device)
    train(net=net,
          trainloader=loader["trainloader"],
          testloader=loader["testloader"])

    if config["train_config"]["save"]:
        _path = "vision/ResNet/pretrain_models/"
        path = _path + \
            "pretrained_resnet18_bottleneck.pth" if config["model_config"][
                "bottleneck"] else _path + "pretrained_resnet18.pth"

        torch.save(net.state_dict(), path)
