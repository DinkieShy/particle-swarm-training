import numpy as np

class ParticleSwarm():
    def __init__(self, dim):
        self.count = 3
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
                self.position[dim] = np.random.randint(dimensions[dim][0], dimensions[dim][1])
            else:
                self.position[dim] = np.random.uniform(dimensions[dim][0], dimensions[dim][1])


def main():
    dimensions = {
        "learningRate0": (0.0005, 0.002),
        "learningRate0Epochs": (2, 10),
        "learningRate1": (0.00001, 0.0005),
        "momentum": (0.75, 0.95),
        "weightDecay": (0.00001, 0.0001)
    }

    swarm = ParticleSwarm(dimensions)

    swarm.initialiseSwarm()

    print(swarm.particles[0].position["learningRate0Epochs"])

if __name__ == "__main__":
    main()