import torch
import torch.nn as nn
from model import ResNet
from utils import train, load_cifar10
from config import config

seed = 1297978
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    loader = load_cifar10(batch_size=config["bath_size"])
    net = ResNet(**config["model"]).to(device)
    train(net=net,
          config=config,
          trainloader=loader["trainloader"],
          testloader=loader["testloader"])
