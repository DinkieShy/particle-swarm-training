import numpy as np
import subprocess
from multiprocessing import Pool, TimeoutError
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

class ParticleSwarm():
    def __init__(self, dim, count=3):
        self.count = count
        self.dimensions = dim # 2d dict of [dimension][0|1] for lower/upper bound
        '''
        Learning rate
        Momentum
        Weight Decay
        '''
        self.particles = []

    def initialiseSwarm(self):
        for i in range(self.count):
            self.particles.append(Particle(self.dimensions))


class Particle():
    def __init__(self, dimensions):
        self.position = {} # dict of [dimension][value]
        self.velocity = {}
        self.momentum = 1

        for dim in dimensions:
            if isinstance(dimensions[dim][0], (int)):
                self.position[dim] = str(np.random.randint(dimensions[dim][0], dimensions[dim][1]))
            else:
                self.position[dim] = str(np.random.uniform(dimensions[dim][0], dimensions[dim][1]))

def runParticle(particle, progBar = None):
    args = ["python", "runNetwork.py"]
    for (key, value) in particle.position.items():
        args.append(key)
        args.append(value)
    
    output = subprocess.run(args, capture_output=True)
    if progBar is not None:
        progBar.update()
    return float(output.stdout)

def main():
    MAX_THREADS = 5

    dimensions = {
        "--lr": (0.0005, 0.002),
        "--lr-drop": (2, 10),
        "--lr2": (0.00001, 0.0005),
        "--momentum": (0.75, 0.95),
        "--decay": (0.00001, 0.0001),
        "--batch-size": (16, 128)
    }

    swarm = ParticleSwarm(dimensions, count=30)

    swarm.initialiseSwarm()
    output = np.array([], dtype=np.float32)

    with Pool(processes=MAX_THREADS) as pool:
        try:
            pbar = tqdm(total=swarm.count, miniters=1, mininterval=1, maxinterval=1)
            for run in pool.map(runParticle, swarm.particles):
                output = np.append(output, [run])
                pbar.update()
        except TimeoutError:
            assert False, "Timeout error"

    # output = process_map(runParticle, swarm.particles, max_workers=MAX_THREADS, total=swarm.count, monitor_interval=1)

    print(output)
    print(f'max: {output.max()}, min: {output.min()}, mean: {output.mean()}, median: {np.median(output)}')


if __name__ == "__main__":
    main()