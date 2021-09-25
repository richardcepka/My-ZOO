import torch
import torch.nn as nn
from model import ViT
from vision.utils import train, load_cifar10
from config import config

seed = 1297978
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    loader = load_cifar10(batch_size=config["bath_size"])
    net = ViT(**config["model"]).to(device)
    train(net=net,
          config=config,
          device=device,
          trainloader=loader["trainloader"],
          testloader=loader["testloader"])
