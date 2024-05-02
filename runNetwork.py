import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from math import isfinite
import os
import random

import argparse
from sys import argv

from darknet import Darknet
from datasets.beetData import AugmentedBeetDataset
from datasets import CustomTransforms as customTransforms
from utils import collate_fn

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from tqdm import tqdm
# import time

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
    
def train(args, model, device, train_loader, optimizer, batchSize, epoch, gradClip, log):
    model.train() 
    runningloss = 0
    optimizer.zero_grad()
    for itr, (dataBatch, targets) in enumerate(pbar := tqdm(train_loader)):
    # for itr, (dataBatch, targets) in enumerate(train_loader):
        data = dataBatch[0].unsqueeze(0)
        for i in range(1, len(dataBatch)):
            data = torch.cat((data, dataBatch[i].unsqueeze(0)), dim=0)
        data = data.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if args.network == "darknet":
            from darknet import computeLoss
            model.losses = []
            output = model(data, targets, CUDA=torch.cuda.is_available())
            loss, losses = computeLoss(output, targets, model)
            if not isfinite(loss.item()):
                return loss.item()
            runningloss += loss.item()
            pbar.set_postfix({'loss': f"{(runningloss/(itr+1)):0.4f}"})
            if log:
                with open("./trainingLog.txt", "a") as f:
                    f.write(f"{epoch}: ({losses[0].item()},{losses[1].item()},{losses[2].item()})\n")
                    f.close()
        elif args.network == "fasterrcnn":
            loss = model(data, targets)
            pbar.set_postfix({'loss': f"{(runningloss/(itr+1)):0.4f}"})
            # pbar.set_postfix({'loss': f"{loss.item():0.4f}"})
            if log:
                with open("./trainingLog.txt", "a") as f:
                    f.write(f'{epoch}: ({loss["loss_box_reg"].item()},{loss["loss_classifier"].item()},{loss["loss_objectness"].item()})\n')
                    f.close()
            loss = sum([val for val in loss.values()])
            runningloss += loss.item()
        else:
            output = model(data)
            loss = F.nll_loss(output, targets)    

        # print(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),gradClip)
        if (itr % batchSize == 0 and itr > 0) or itr+1 == len(train_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return runningloss / len(train_loader)

def test(model, test_loader, device):
    model.eval()

    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for (image, label) in test_loader:
            image, label = image.to(device), label.to(device)
            prediction = model(image)
            correct += (prediction.argmax(1) == label).type(torch.float).sum().item()

    print(f"Correct percentage: {(100*(correct/total)):>0.2f}%")

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Network Training')
parser.add_argument("--training-batch", type=int, default=2, metavar="B",
                    help="input inputs to train at a time (default: 2)")
parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                    help='number of training batches to step optimiser (default: 12)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lr2', type=float, default=0.0001, metavar='LR2',
                    help="second learning rate to use")
parser.add_argument('--lr-drop', type=int, default=10, metavar="LRD",
                    help="how many epochs to wait before dropping learning rate")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help="momentum value for SGD")
parser.add_argument('--decay', type=float, default=0, metavar='WD',
                    help="weight decay value")
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test', type=bool, default=False, metavar='T',
                    help='run on test set (default: False)')
parser.add_argument("--network", type=str, default="darknet", metavar="network",
                    help="Network to use (default: \"darknet\")")
parser.add_argument("--grad-clip", type=float, default=0.5, metavar="x",
                    help="clip gradients to x during training (default 0.5)")
parser.add_argument("--log", type=bool, default=False,
                    help="output training losses to ./trainingLog.txt (default False)")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = True

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': args.training_batch,
                    'shuffle': True}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True}
    train_kwargs.update(cuda_kwargs)


def transform(image, targets):
    if image.size != (800, 1216):
        image, targets["boxes"] = customTransforms.resize(image, targets["boxes"], (800,1216))
    image = transforms.ToTensor()(image)
    image = F.normalize(image)
    return image, targets

trainDataset = AugmentedBeetDataset("/datasets/LincolnAugment/train.txt", transform=transform)
train_loader = torch.utils.data.DataLoader(trainDataset, collate_fn=collate_fn, **train_kwargs)

valDataset = AugmentedBeetDataset("/datasets/LincolnAugment/val.txt", transform=transform)
test_loader = torch.utils.data.DataLoader(valDataset, collate_fn=collate_fn, **train_kwargs)

lossAgent = None
numClasses = 2
if args.network == "darknet":
    cfgPath = os.path.abspath("./cfg/yolov3Custom.cfg")
    assert os.path.exists(cfgPath)
    model = Darknet(cfgPath).to(device)
elif args.network == "fasterrcnn":
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=numClasses+1).to(device)
elif args.network == "simplenet":
    model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr)

f = open("./trainingLog.txt", "w")
f.close()

for epoch in range(1, args.epochs+1):
    loss = train(args, model, device, train_loader, optimizer, args.batch_size, epoch, args.grad_clip, args.log)
    if not isfinite(loss):
        break
    # Log training loss to file (only use for testing; will break main.py)
    if epoch == args.lr_drop:
        for i in optimizer.param_groups:
            i['lr'] = args.lr2

print(loss)

if args.test:
    test(model, test_loader, device)
