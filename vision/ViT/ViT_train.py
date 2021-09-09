import torch
import torch.optim as optim
import torch.nn as nn
from ViT_model import vit
from transformers import get_cosine_schedule_with_warmup
from vit_utils import acc, evaluate, load_cifar10
from vit_config import config

seed = 1297978
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(net, trainloader, testloader=None):
    """
    train: Adam, cosine schedule with linear warm up, gradient cliping = False
    """

    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(), lr=config["train_config"]["lr"], weight_decay=config["train_config"]["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=config["train_config"]["warmup_epochs"], num_training_steps=config["train_config"]["epochs"])

    for epoch in range(1, config["train_config"]["epochs"]+1):
        train_loss = 0
        train_acc = 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            if config["train_config"]["gradient_clipping"]:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
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
    loader = load_cifar10()
    net = vit(clasification_head=config["model_config"]["clasification_head"]).to(
        device)
    train(net=net,
          trainloader=loader["trainloader"],
          testloader=loader["testloader"])

    if config["train_config"]["save"]:
        _path = "vision/ViT/pretrain_models/"
        path = _path + \
            "pretrained_vit_cls.pth" if config["model_config"]["clasification_head"] == "cls" else _path + \
            "vision/ViT/pretrain_models/pretrained_vit_mean.pth"

        torch.save(net.state_dict(), path)
