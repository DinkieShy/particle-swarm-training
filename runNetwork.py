import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from math import isfinite
import os

import argparse
from sys import argv

from darknet import YoloLoss, Darknet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
def train(args, model, device, train_loader, optimizer, epoch, lossAgent = None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        match args.network:
            case "darknet":
                output = model(data, CUDA=torch.cuda.is_available())
                losses = [[] for _ in range(7)]
                # This is weird but useful when detecting at different scales
                for i, lossItem in enumerate(lossAgent(output, target)):
                    losses[i].append(lossItem)
                losses = [sum(i) for i in losses]
                loss = losses[0]

            case _:
                output = model(data)
                loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        
    return loss

def test(model, test_loader, device):
    model.eval()

    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            correct += (prediction.argmax(1) == label).type(torch.float).sum().item()

    print(f"Correct percentage: {(100*(correct/total)):>0.2f}%")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--lr2', type=float, default=0.0001, metavar='LR2',
                    help="second learning rate to use")
parser.add_argument('--lr-drop', type=int, default=10, metavar="LRD",
                    help="how many epochs to wait before dropping learning rate")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help="momentum value for SGD")
parser.add_argument('--decay', type=float, default=0.00001, metavar='WD',
                    help="weight decay value")
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test', type=bool, default=False, metavar='T',
                    help='run on test set (default: False)')
parser.add_argument("--network", type=str, default="darknet", metavar="network",
                    help="Network to use (default: \"darknet\")")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': args.batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((416, 416), antialias=False)
    # transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.OxfordIIITPet('./data', download=True,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)

dataset2 = datasets.OxfordIIITPet('./data', download=True,
                    transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2,**train_kwargs)

lossAgent = None
numClasses = 2
match args.network:
    case "darknet":
        cfgPath = os.path.abspath("./cfg/yolov3.cfg")
        assert os.path.exists(cfgPath)
        model = Darknet(cfgPath)
        lossAgent = YoloLoss(18, numClasses, (model.net_info["width"], model.net_info["height"]))

    case "simplenet":
        model = Net().to(device)

optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

for epoch in range(1, 15 + 1):
    loss = train(args, model, device, train_loader, optimizer, epoch, lossAgent)
    if epoch == args.lr_drop:
        for i in optimizer.param_groups:
            i['lr'] = args.lr2

if isfinite(loss.item()):
    print(loss.item())
else:
    print("NaN")

if args.test:
    test(model, test_loader, device)