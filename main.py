import numpy as np
import subprocess
from multiprocessing import Pool, TimeoutError, Process
from sys import float_info
from json import dumps
import argparse

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

def runParticle(args, progBar = None):
    executable = args[0].exe
    particle = args[1]
    # on non-docker OS, need to specify python exe
    # args = ["./env/Scripts/python.exe", "runNetwork.py"]
    args = [f"{executable} runNetwork.py --network {args[0].network}"]
    for (key, value) in particle.position.items():
        args.append(key)
        args.append(str(value))
        # print(f"{key}: {str(value)}")

    # creates string like "python runNetwork.py --batch-size 8 --lr 0.0005 --lr-drop 10"
    # then runs it in a terminal and captures the output, converting it to a float and storing
    # runNetwork.py should therefore ONLY output the final score to be used by the particle swarm

    output = subprocess.run(args, capture_output=True)
    if progBar is not None: # Tried to use tqdm progress bar but this tends to break
        progBar.update()
    assert output.stderr == b'', f"Error from subprocess: {output.stderr}"
    result = float(output.stdout)
    particle.update(result) # Calls the particle with it's result to update
    return [result, particle]

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--exe', type=str, default="python", metavar='file',
                    help='python executable to use (default: "python")')
    parser.add_argument("--network", type=str, default="simplenet", metavar="network",
                    help="Network to use (default: \"simplenet\")")
    args = parser.parse_args()

    MAX_THREADS = 2
    MAX_ITERATIONS = 3
    RUNS_PER_ITERATION = 20
    PARTICLES = 10
    SPEED = 0.0001 # initial learning rate of the particles
    MOMENTUM = 0.5 # decay of speed

    dimensions = {
        "--lr": (0.0005, 0.005),
        "--lr-drop": (2, 10),
        "--lr2": (0.00001, 0.0005),
        "--momentum": (0.75, 0.95),
        "--decay": (0.00001, 0.0001),
        "--batch-size": (32, 128)
    }

    favouredPairs = [
        ["--lr", "--lr2"],
        ["--lr2", "--lr-drop"],
        ["--lr", "--lr-drop"],
    ]

    swarm = ParticleSwarm(dimensions, count=PARTICLES, pairs = favouredPairs, SPEED)

    swarm.initialiseSwarm()
    oldParticles = []

    bestResult = -1
    bestPosition = {}

    with Pool(processes=MAX_THREADS) as pool:
        try:
            for run in range(MAX_ITERATIONS):
                output = np.array([], dtype=np.float32)
                print(f"Starting run {run}")
                newParticles = []
                for out in pool.map(runParticle, zip([args for _ in range(PARTICLES)], swarm.particles)): # Run the particles in parallel
                    # Currently, max concurrent threads is just user defined.
                    # possible to estimate memory usage and automatically optimise concurrent thread count?
                    output = np.append(output, [out[0]])
                    if run != 0:
                        oldParticles.append(out[1])
                    newParticles.append(out[1])
                print(f"Run {run} complete")
                if output.min() <= bestResult or bestResult == -1:
                    bestPosition = newParticles[np.nonzero(output == output.min())[0][0]].position
                    bestResult = output.min()
                if run % RUNS_PER_ITERATION == 0:
                    swarm.speed *= MOMENTUM
                    swarm.initialiseSwarm(bestPosition, bestResult) # Initialise next set of particles based on all-time best result
                else:
                    swarm.particles = newParticles
                print(f'max: {output.max()}, min: {output.min()}, mean: {output.mean()}, median: {np.median(output)}')
                
        except TimeoutError:
            assert False, "Timeout error"

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