import numpy as np
import subprocess
from multiprocessing import Pool, TimeoutError, Process
from sys import float_info
from json import dumps

class ParticleSwarm():
    def __init__(self, dim, count=3):
        self.count = count
        self.dimensions = dim # 2d dict of [dimension][0|1] for lower/upper bound
        self.particles = []

    def initialiseSwarm(self, distribution = None, result = None):
        self.particles = []
        if distribution == None:
            for _ in range(self.count):
                self.particles.append(Particle(self.dimensions))
        else:
            for _ in range(self.count):
                self.particles.append(Particle(self.dimensions, distribution, result))



class Particle():
    def __init__(self, dimensions, distribution = None, result = None):
        self.dimensions = dimensions
        self.position = {} # dict of [dimension][value]
        self.velocity = {}
        self.momentum = 0.9
        self.speed = 0.0001
        self.positions = []
        self.results = []
        self.randomGen = np.random.default_rng()

        if distribution != None:
            self.positions.append(distribution)
        if result != None:
            self.results.append(result)

        for dim in self.dimensions:
            if isinstance(self.dimensions[dim][0], (int)):
                if distribution == None:
                    self.position[dim] = np.random.randint(self.dimensions[dim][0], self.dimensions[dim][1])
                else:
                    self.position[dim] = round(self.randomGen.normal(distribution[dim]/float(self.dimensions[dim][1]), 0.25)*self.dimensions[dim][1])
            else:
                if distribution == None:
                    self.position[dim] = np.random.uniform(self.dimensions[dim][0], self.dimensions[dim][1])
                else:
                    self.position[dim] = self.randomGen.normal(distribution[dim]/self.dimensions[dim][1], 0.25)*self.dimensions[dim][1]

            if self.position[dim] > max(self.dimensions[dim]):
                self.position[dim] = max(self.dimensions[dim])
            elif self.position[dim] < min(self.dimensions[dim]):
                self.position[dim] = min(self.dimensions[dim])

    def intCheck(self):
        # parameters that are integers should stay integers
        for dim in self.dimensions:
            if isinstance(self.dimensions[dim][0], (int)):
                self.position[dim] = round(self.position[dim])

    def update(self, result):
        newPosition = {}
        for dim in self.position:
            normalisationFactor = float(self.dimensions[dim][1])
            currentValue = (self.position[dim], result)
            lastValue = (self.positions[-1][dim], self.results[-1]) if len(self.results) > 0 else (1, 1)

            gradient = (currentValue[1]-lastValue[1])/(currentValue[0]-lastValue[0]+float_info.epsilon)

            newPosition[dim] = (currentValue[0] - gradient*self.speed)

            if newPosition[dim] > max(self.dimensions[dim]):
                newPosition[dim] = max(self.dimensions[dim])
            elif newPosition[dim] < min(self.dimensions[dim]):
                newPosition[dim] = min(self.dimensions[dim])

        self.speed *= self.momentum
        self.positions.append(self.position)
        self.results.append(result)
        self.position = newPosition

        self.intCheck()
        print(self.results)

    def maxReasonableVelocity(self, dimension):
        # velocity should be limited based on range of dimension
        velRange = max(dimension) - min(dimension)
        return 0.3 * velRange

def runParticle(particle, progBar = None):
    args = ["python", "runNetwork.py"]
    for (key, value) in particle.position.items():
        args.append(key)
        args.append(str(value))
        # print(f"{key}: {str(value)}")

    output = subprocess.run(args, capture_output=True)
    if progBar is not None:
        progBar.update()
    assert output.stderr == b'', f"Error from subprocess: {output.stderr}"
    result = float(output.stdout)
    particle.update(result)
    return [result, particle]

def main():
    MAX_THREADS = 5
    MAX_RUNS = 50
    PARTICLES = 5

    dimensions = {
        "--lr": (0.0005, 0.005),
        "--lr-drop": (2, 10),
        "--lr2": (0.00001, 0.0005),
        "--momentum": (0.75, 0.95),
        "--decay": (0.00001, 0.0001),
        "--batch-size": (32, 128)
    }

    swarm = ParticleSwarm(dimensions, count=PARTICLES)

    swarm.initialiseSwarm()
    oldParticles = []

    with Pool(processes=MAX_THREADS) as pool:
        try:
            for run in range(MAX_RUNS):
                output = np.array([], dtype=np.float32)
                print(f"Starting run {run}")
                newParticles = []
                for out in pool.map(runParticle, swarm.particles):
                    output = np.append(output, [out[0]])
                    if run % 5 == 0:
                        oldParticles.append(out[1])
                    newParticles.append(out[1])
                print(f"Run {run} complete")
                bestResult = output.min()
                if run % 5 == 0:
                    swarm.initialiseSwarm(newParticles[np.nonzero(output == bestResult)[0][0]].position, bestResult)
                else:
                    swarm.particles = newParticles
                print(f'max: {output.max()}, min: {bestResult}, mean: {output.mean()}, median: {np.median(output)}')
                
        except TimeoutError:
            assert False, "Timeout error"

    particleResults = []
    for i in oldParticles:
        result = {}
        for ii in range(len(i.results)):
            result[ii] = [i.positions[ii], i.results[ii]]
        particleResults.append(result)
    
    with open("./result.json", "w") as file:
        file.write(dumps(particleResults))

if __name__ == "__main__":
    main()