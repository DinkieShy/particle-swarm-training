import numpy as np
from sys import float_info
from json import dumps
from math import isfinite
import argparse
from runNetwork import train
import torch
import random
from darknet import Darknet
from datasets.beetData import AugmentedBeetDataset
from datasets import CustomTransforms as customTransforms
from utils import collate_fn
import os

import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


# Install: pip install numpy torch torchvision

class ParticleSwarm():
    def __init__(self, dim, count=3, pairs=[], speed=0.0001):
        self.count = count
        self.dimensions = dim # 2d dict of [dimension][0|1] for lower/upper bound
        self.particles = []
        self.randomGen = np.random.default_rng()
        self.speed = speed
        self.favouredPairs = pairs

    def initialiseSwarm(self, distribution = None, result = None):
        pairsToChange = self.favouredPairs
        while len(pairsToChange) < self.count:
            newPair = [list(self.dimensions.keys())[self.randomGen.integers(0, len(self.dimensions))] for _ in range(2)]
            unique = False
            maxTries = 5
            tries = 0
            while not unique and tries <= maxTries:
                for i in pairsToChange:
                    if newPair[0] in i and newPair[1] in i:
                        newPair = [list(self.dimensions.keys())[self.randomGen.integers(0, len(self.dimensions))] for _ in range(2)]
                        tries += 1
                        break
                unique = True
            pairsToChange.append(newPair)

        self.particles = []
        if distribution == None:
            for _ in range(self.count):
                self.particles.append(Particle(hex(int(self.randomGen.random()*131064)), self.dimensions, int(self.randomGen.random()*1000000), self.speed, pairsToChange.pop()))
        else:
            for _ in range(self.count):
                self.particles.append(Particle(hex(int(self.randomGen.random()*131064)), self.dimensions, int(self.randomGen.random()*1000000), self.speed, pairsToChange.pop(), distribution, result))

class Particle():
    def __init__(self, idString, dimensions, randomSeed, speed, dimensionsToChange, distribution = None, result = None):
        self.id = idString
        self.dimensions = dimensions
        self.position = {} # dict of [dimension][value]
        self.velocity = {}
        self.speed = speed
        self.momentum = 0.9
        self.positions = []
        self.results = []
        self.randomGen = np.random.default_rng(seed=randomSeed)

        if distribution != None:
            self.positions.append(distribution)
        if result != None:
            self.results.append(result)

        self.dimensionsBeingChanged = dimensionsToChange
        self.setRandomPosition(distribution)
    
    def setRandomPosition(self, distribution = None):
        for dim in self.dimensions:
            if dim in self.dimensionsBeingChanged or distribution == None:
                if isinstance(self.dimensions[dim][0], (int)):
                    if distribution == None:
                        self.position[dim] = np.random.randint(self.dimensions[dim][0], self.dimensions[dim][1])
                    else:
                        self.position[dim] = round(self.randomGen.normal(distribution[dim]/float(self.dimensions[dim][1]), 0.15)*self.dimensions[dim][1])
                else:
                    if distribution == None:
                        self.position[dim] = np.random.uniform(self.dimensions[dim][0], self.dimensions[dim][1])
                    else:
                        self.position[dim] = self.randomGen.normal(distribution[dim]/self.dimensions[dim][1], 0.15)*self.dimensions[dim][1]

                if self.position[dim] > max(self.dimensions[dim]):
                    self.position[dim] = max(self.dimensions[dim])
                elif self.position[dim] < min(self.dimensions[dim]):
                    self.position[dim] = min(self.dimensions[dim])
            else:
                self.position[dim] = distribution[dim]

    def intCheck(self):
        # parameters that are integers should stay integers
        for dim in self.dimensions:
            if isinstance(self.dimensions[dim][0], (int)):
                self.position[dim] = round(self.position[dim])

    def update(self, result):
        newPosition = {}
        for dim in self.position:
            if dim in self.dimensionsBeingChanged:
                normalisationFactor = float(self.dimensions[dim][1]) # Normalise to prevent the result being changed by too high a factor
                currentValue = (self.position[dim]/normalisationFactor, result)
                lastValue = (self.positions[-1][dim]/normalisationFactor, self.results[-1]) if len(self.results) > 0 else (1, 1)

                gradient = (currentValue[1]-lastValue[1])/(currentValue[0]-lastValue[0]+float_info.epsilon) # Simple y dif / x dif gradient from last point

                newPosition[dim] = (currentValue[0] - gradient*self.speed)*normalisationFactor # Update value based on gradient and decaying speed

                if newPosition[dim] > max(self.dimensions[dim]): # Crop to be within limits
                    newPosition[dim] = max(self.dimensions[dim])
                elif newPosition[dim] < min(self.dimensions[dim]):
                    newPosition[dim] = min(self.dimensions[dim])
            else:
                newPosition[dim] = self.position[dim] # If value shouldn't be changed in this particle, ignore and move on

        self.speed *= self.momentum # Decay speed

        self.positions.append(self.position) # Keep track of positions/results
        self.results.append(result)
        self.position = newPosition

        self.intCheck()

    def maxReasonableVelocity(self, dimension):
        # velocity should be limited based on range of dimension
        velRange = max(dimension) - min(dimension)
        return 0.3 * velRange
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--exe', type=str, default="python", metavar='file',
                    help='python executable to use (default: "python")')
    parser.add_argument("--network", type=str, default="simplenet", metavar="network",
                    help="Network to use (default: \"simplenet\")")
    args = parser.parse_args()

    MAX_ITERATIONS = 15 # This * particles is how many networks get trained
    RUNS_PER_ITERATION = 15 # Reinit particles every _ runs
    PARTICLES = 5
    SPEED = 0.0001 # initial learning rate of the particles
    MOMENTUM = 0.5 # decay of speed

    TRAINING_BATCH = 3
    BATCH_SIZE=20
    SEED = 1
    EPOCHS = 10

    dimensions = {
        "lr": (0.001, 0.1),
        "lr_drop": (1, 3),
        "lr2": (0.00001, 0.001),
        "grad_clip": (0.05, 5)
        # "batch_size": (4, 64)
        # "momentum": (0.75, 0.95),
        # "decay": (0.00001, 0.0001)
    }

    favouredPairs = [
        ["lr", "lr2"],
        ["lr2", "lr_drop"],
        ["lr", "lr_drop"],
        ["lr", "grad_clip"]
    ]

    swarm = ParticleSwarm(dimensions, count=PARTICLES, pairs = favouredPairs, speed=SPEED)

    swarm.initialiseSwarm()
    oldParticles = []

    bestResult = -1
    bestPosition = {}

    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    train_kwargs = {'batch_size': TRAINING_BATCH,
                        'shuffle': True}

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
    else:
        device = torch.device("cpu")

    def transform(image, targets):
        if image.size != (800, 1216):
            image, targets["boxes"] = customTransforms.resize(image, targets["boxes"], (800,1216))
        image = transforms.ToTensor()(image)
        image = F.normalize(image)
        return image, targets

    trainDataset = AugmentedBeetDataset("/datasets/LincolnAugment/train.txt", transform=transform)
    train_loader = torch.utils.data.DataLoader(trainDataset, collate_fn=collate_fn, **train_kwargs)

    # valDataset = AugmentedBeetDataset("/datasets/LincolnAugment/val.txt", transform=transform)
    # test_loader = torch.utils.data.DataLoader(valDataset, collate_fn=collate_fn, **train_kwargs)

    for run in range(MAX_ITERATIONS):
        output = np.array([], dtype=np.float32)
        print(f"Starting run {run}")
        output = []
        for i in range(PARTICLES):
            gradClip = swarm.particles[i].position["grad_clip"]
            numClasses = 2
            if args.network == "darknet":
                cfgPath = os.path.abspath("./cfg/yolov3Custom.cfg")
                assert os.path.exists(cfgPath)
                model = Darknet(cfgPath).to(device)
            elif args.network == "fasterrcnn":
                model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=numClasses+1).to(device)
            # elif args.network == "simplenet":
            #     model = Net().to(device)
            optimizer = optim.SGD(model.parameters(), lr=swarm.particles[i].position["lr"])
            for epoch in range(1, EPOCHS+1):
                assert torch.cuda.is_available(), "...what?"
                loss = train(args.network, model, device, train_loader, optimizer, BATCH_SIZE, epoch, gradClip, False)
                if not isfinite(loss):
                    break
                if epoch == swarm.particles[i].position["lr_drop"]:
                    for x in optimizer.param_groups:
                        x['lr'] = swarm.particles[i].position["lr2"]
            del model
            del optimizer
            print(loss)
            swarm.particles[i].positions.append(swarm.particles[i].position)
            swarm.particles[i].update(result)
            output.append(loss)
            
        print(f"Run {run} complete")
        if output.min() <= bestResult or bestResult == -1:
            bestPosition = swarm.particles[np.argmin(output)].position
            bestResult = output.min()
        if run % RUNS_PER_ITERATION == 0:
            swarm.speed *= MOMENTUM
            swarm.initialiseSwarm(bestPosition, bestResult) # Initialise next set of particles based on all-time best result
        print(f'max: {output.max()}, min: {output.min()}, mean: {output.mean()}, median: {np.median(output)}')
        particleResults = []
        for i in oldParticles:
            result = {}
            for ii in range(len(i.results)):
                result[i.id] = [i.positions[ii], i.results[ii]]
            particleResults.append(result)
        
        with open("./swarmResult.json", "w") as file:
            file.write(dumps(particleResults))

if __name__ == "__main__":
    main()
