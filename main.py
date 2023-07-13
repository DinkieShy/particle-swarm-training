import numpy as np
import subprocess
from multiprocessing import Pool, TimeoutError, Process
from json import dumps

class ParticleSwarm():
    def __init__(self, dim, count=3):
        self.count = count
        self.dimensions = dim # 2d dict of [dimension][0|1] for lower/upper bound
        self.particles = []

    def initialiseSwarm(self):
        for _ in range(self.count):
            self.particles.append(Particle(self.dimensions))


class Particle():
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.position = {} # dict of [dimension][value]
        self.velocity = {}
        self.momentum = 0.8
        self.positions = []
        self.results = []
        self.changeThreshold = 0.2

        for i in self.dimensions:
            self.velocity[i] = np.random.random()*self.maxReasonableVelocity(self.dimensions[i])
            if np.random.random() > 0.5:
                self.velocity[i] *= -1

        for dim in self.dimensions:
            if isinstance(self.dimensions[dim][0], (int)):
                self.position[dim] = np.random.randint(self.dimensions[dim][0], self.dimensions[dim][1])
            else:
                self.position[dim] = np.random.uniform(self.dimensions[dim][0], self.dimensions[dim][1])

    def intCheck(self):
        # parameters that are integers should stay integers
        for dim in self.dimensions:
            if isinstance(self.dimensions[dim][0], (int)):
                self.position[dim] = np.round(self.position[dim])

    def update(self, result):
        self.results.append(result)
        self.positions.append(self.position)
        print(self.results)
        if len(self.results) >= 2:
            # After a couple runs, check if the result is improving
            # If it's getting worse by a threshold, each velocity comp gets a 30% to be reinitialised
            resultTheta = self.results[-1] - self.results[-2]
            if resultTheta > self.changeThreshold:
                # result is training loss, so going *up* is bad
                for i in self.velocity:
                    if np.random.random() <= 0.3:
                        self.velocity[i] = np.random.random()*self.maxReasonableVelocity(self.dimensions[i])
                        if np.random.random() > 0.5:
                            self.velocity[i] *= -1

        for i in self.velocity:
            self.position[i] += self.velocity[i]*(0.8+np.random.random()*0.4)
            if self.position[i] > max(self.dimensions[i]):
                self.position[i] = max(self.dimensions[i])
                self.velocity[i] *= -1
            elif self.position[i] > min(self.dimensions[i]):
                self.position[i] = min(self.dimensions[i])
                self.velocity[i] *= -1

            self.velocity[i] *= self.momentum

        self.momentum *= self.momentum

        self.intCheck()

    def maxReasonableVelocity(self, dimension):
        # velocity should be limited based on range of dimension
        velRange = max(dimension) - min(dimension)
        return 0.3 * velRange

def runParticle(particle, progBar = None):
    args = ["python", "runNetwork.py"]
    for (key, value) in particle.position.items():
        args.append(key)
        args.append(str(value))
    
    output = subprocess.run(args, capture_output=True)
    if progBar is not None:
        progBar.update()
    result = float(output.stdout)
    particle.update(result)
    return [result, particle]

def main():
    MAX_THREADS = 5
    MAX_RUNS = 5
    PARTICLES = 10

    dimensions = {
        "--lr": (0.0005, 0.002),
        "--lr-drop": (2, 10),
        "--lr2": (0.00001, 0.0005),
        "--momentum": (0.75, 0.95),
        "--decay": (0.00001, 0.0001),
        "--batch-size": (16, 128)
    }

    swarm = ParticleSwarm(dimensions, count=PARTICLES)

    swarm.initialiseSwarm()

    with Pool(processes=MAX_THREADS) as pool:
        try:
            for run in range(MAX_RUNS):
                output = np.array([], dtype=np.float32)
                newParticles = []
                print(f"Starting run {run}")
                for out in pool.map(runParticle, swarm.particles):
                    output = np.append(output, [out[0]])
                    newParticles.append(out[1])
                print(f"Run {run} complete")
                swarm.particles = newParticles
                # results = np.array([i.results[-1] for i in swarm.particles])
                print(f'max: {output.max()}, min: {output.min()}, mean: {output.mean()}, median: {np.median(output)}')
                
        except TimeoutError:
            assert False, "Timeout error"

    particleResults = []
    for i in swarm.particles:
        result = {}
        for ii in range(len(i.results)):
            result[ii] = [i.positions[ii], i.results[ii]]
    
    with open("./result.json", "w") as file:
        file.write(dumps(particleResults))

if __name__ == "__main__":
    main()