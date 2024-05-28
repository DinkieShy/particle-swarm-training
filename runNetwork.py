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
    
def train(network, model, device, train_loader, optimizer, batchSize, epoch, gradClip, log):
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
        if network == "darknet":
            from darknet import computeLoss
            model.losses = []
            output = model(data, CUDA=torch.cuda.is_available())
            loss, losses = computeLoss(output, targets, model)
            if not isfinite(loss.item()):
                return loss.item()
            runningloss += loss.item()
            pbar.set_postfix({'loss': f"{(runningloss/(itr+1)):0.4f}"})
            if log:
                with open("./trainingLog.txt", "a") as f:
                    f.write(f"{epoch}: ({losses[0].item()},{losses[1].item()},{losses[2].item()})\n")
                    f.close()
        elif network == "fasterrcnn":
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

def test(model, test_loader, device, network, numClasses, IOU_THRESH=0.5, CONF_THRESH=0.5):
    print("Starting test")
    # IOU_THRESH = IoU score to count as true positive
    # CONF_THRESH = Confidence threshold below which predictions are ignored
    model.eval()
    truePos = [0 for _ in range(numClasses)]
    falsePos = [0 for _ in range(numClasses)]
    falseNeg = [0 for _ in range(numClasses)]
    with torch.no_grad():
        for (dataBatch, targets) in tqdm(test_loader):
            data = dataBatch[0].unsqueeze(0)
            for i in range(1, len(dataBatch)):
                data = torch.cat((data, dataBatch[i].unsqueeze(0)), dim=0)
            data = data.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            if network == "darknet":
                outputs = model(data, CUDA=torch.cuda.is_available())
                # outputs is in shape [scale][batch size, detections, bboxAttributes]
                detections = []
                for scale in range(len(outputs)):
                    # filter by objectness score above conf thresh
                    for image in range(data.shape[0]):
                        confInd = outputs[scale][image,:,4] >= CONF_THRESH
                        if len(detections) <= image:
                            detections.append(outputs[scale][image,confInd])
                        else:
                            detections[image] = torch.cat((detections[image], outputs[scale][image,confInd]))
                for image in range(data.shape[0]):
                    # convert to x1, y1, x2, y2, class
                    predBoxes = torch.zeros_like(detections[image][...,:4], device=device)
                    predBoxes[...,0:2] = detections[image][...,0:2] - detections[image][...,2:4]/2 # x, y -> x1, y1
                    predBoxes[...,2:4] = detections[image][...,0:2] + detections[image][...,2:4]/2 # w, h -> x2, y2
                    classes = torch.argmax(detections[image][...,5:], dim=-1)
                    confidences = torch.cat((detections[image][...,4:5], detections[image][classes+5]), dim=1)
                    detections[image] = torch.cat((predBoxes, classes.unsqueeze(1), confidences), dim=1) # This removes confidence scores
                for image in range(data.shape[0]):
                    shape = list(detections[image].shape)
                    shape[-1] =2
                    for box in range(len(targets[image]["boxes"])):
                        targetBox = targets[image]["boxes"][box]
                        intersects = torch.zeros(shape, device=device)
                        invalidIntersects = 	torch.stack((detections[image][...,2] > targetBox[2], detections[image][...,0] < targetBox[0],
                                                            detections[image][...,3] > targetBox[3], detections[image][...,1] < targetBox[1]),dim=-1)
                        invalidIntersects = torch.any(invalidIntersects, -1)
                        intersects[...,0] = torch.minimum(detections[image][...,2], targetBox[2])-torch.maximum(detections[image][...,0],targetBox[0])
                        intersects[...,1] = torch.minimum(detections[image][...,3], targetBox[3])-torch.maximum(detections[image][...,1],targetBox[1])
                        intersects = torch.prod(torch.clamp(intersects,min=0), dim=-1) # This is now the intersection AREAS
                        intersects[~invalidIntersects] = 0
                        areas = (detections[image][...,2] - detections[image][...,0])*(detections[image][...,3] - detections[image][...,1])
                        targetArea = (targetBox[2]-targetBox[0])*(targetBox[3]-targetBox[1])
                        ious = intersects / (areas+targetArea-intersects)
                        iouMatch = ious >= IOU_THRESH
                        classMatch = detections[image][...,5] == targets[image]["labels"][box]-1
                        if any(torch.logical_and(iouMatch,classMatch)):
                            truePos[targets[image]["labels"][box]-1] += 1
                        elif any(iouMatch):
                            falsePos[targets[image]["labels"][box]-1] += 1
                        else:
                            falseNeg[targets[image]["labels"][box]-1] += 1
    return truePos, falsePos, falseNeg

def main():
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
    parser.add_argument("--grad-clip", type=float, default=1, metavar="x",
                        help="clip gradients to x during training (default 0.5)")
    parser.add_argument("--log", type=bool, default=False,
                        help="output training losses to ./trainingLog.txt (default False)")
    parser.add_argument("--save", type=str, default="", metavar="file path",
                        help="Set to filename of path to save weights to. Leave blank to not save weights")
    parser.add_argument("--load", type=str, default="", metavar="file path",
                        help="Set to filename of path to load weights from. Leave blank to not load weights")
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

    # trainDataset = AugmentedBeetDataset("/datasets/LincolnAugment/trainNonAugment.txt", transform=transform)
    trainDataset = AugmentedBeetDataset("/datasets/LincolnAugment/val.txt", transform=transform)
    train_loader = torch.utils.data.DataLoader(trainDataset, collate_fn=collate_fn, **train_kwargs)

    valDataset = AugmentedBeetDataset("/datasets/LincolnAugment/val.txt", transform=transform)
    test_loader = torch.utils.data.DataLoader(valDataset, collate_fn=collate_fn, **train_kwargs)

    numClasses = 2
    if args.network == "darknet":
        cfgPath = os.path.abspath("./cfg/yolov3Custom.cfg")
        assert os.path.exists(cfgPath)
        model = Darknet(cfgPath).to(device)
    elif args.network == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=numClasses+1).to(device)
    elif args.network == "simplenet":
        model = Net().to(device)

    if args.load != "":
        model.load_state_dict(torch.load(args.load, map_location=device))

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    f = open("./trainingLog.txt", "w")
    f.close()

    for epoch in range(1, args.epochs+1):
        loss = train(args.network, model, device, train_loader, optimizer, args.batch_size, epoch, args.grad_clip, args.log)
        if not isfinite(loss):
            break
        # Log training loss to file (only use for testing; will break main.py)
        if epoch == args.lr_drop:
            for i in optimizer.param_groups:
                i['lr'] = args.lr2

    # print(loss)
    if args.save != "":
        torch.save(model.state_dict(), args.save)

    if args.test:
        truePos, falsePos, falseNeg = test(model, test_loader, device, args.network, numClasses)
        for label in range(numClasses):
            precision = truePos[label] / (truePos[label]+falsePos[label])if truePos[label] > 0 or falsePos[label] > 0 else 0.0
            recall = truePos[label] / (truePos[label] + falseNeg[label]) if truePos[label] > 0 or falseNeg[label] > 0 else 0.0
            print(f"Label: {label}\t\tPrecision: {precision}, Recall: {recall}")

if __name__ == "__main__":
    main()