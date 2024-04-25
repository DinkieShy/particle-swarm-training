import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("agg")

file = "./trainingLog.txt"
assert os.path.isfile(file), "trainingLog not found"

lines = []
with open(file) as f:
    lines = f.readlines()
    f.close()

colours = ["b", "y"]

minEpoch = 0
maxEpoch = int(lines[-1].split(": ")[0])-1
losses = [[] for _ in range(maxEpoch+1)]
for line in lines:
    epoch, loss = line.split(": ")
    epoch = int(epoch)
    loss = float(loss)
    losses[epoch-1].append(loss)

plt.figure(figsize=(19.2,10.8))

xMarkers = []
for epoch in range(maxEpoch+1):
    iterationsOffset = epoch*len(losses[epoch])
    xMarkers.append(iterationsOffset)
    iterations = [x+iterationsOffset for x in range(len(losses[epoch]))]
    plt.plot(iterations, losses[epoch], c=colours[epoch%2])

plt.ylim(0, 10000)
plt.xticks(xMarkers)
plt.savefig("./trainingGraph.png")