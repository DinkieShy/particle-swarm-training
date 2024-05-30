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
bboxLosses = [[] for _ in range(maxEpoch+1)]
clsLosses = [[] for _ in range(maxEpoch+1)]
objLosses = [[] for _ in range(maxEpoch+1)]
for line in lines:
    epoch, loss = line.split(": ")
    epoch = int(epoch)
    loss = loss[1:-2].split(",")
    bboxLosses[epoch-1].append(float(loss[0]))
    clsLosses[epoch-1].append(float(loss[1]))
    objLosses[epoch-1].append(float(loss[2]))

plt.figure(figsize=(19.2,10.8))

xMarkers = []
bboxAxBig = plt.subplot(2,2,3)
bboxAxSmol = plt.subplot(2,2,4)
clsAx = plt.subplot(2,2,1)
objAx = plt.subplot(2,2,2)
bboxAxBig.set_title("Bounding box loss (MSE)")
bboxAxSmol.set_title("Bounding box loss (MSE)")
clsAx.set_title("Classification loss (BCE)")
objAx.set_title("Object Confidence loss (BCE)")
# bboxAxBig.set_ylim(0,100)
bboxAxSmol.set_ylim(0,1)

for epoch in range(maxEpoch+1):
    iterationsOffset = epoch*len(bboxLosses[0])
    xMarkers.append(iterationsOffset)
    iterations = [x+iterationsOffset for x in range(len(bboxLosses[epoch]))]
    bboxAxBig.plot(iterations, bboxLosses[epoch], c=colours[epoch%2])
    bboxAxSmol.plot(iterations, bboxLosses[epoch], c=colours[epoch%2])
    clsAx.plot(iterations, clsLosses[epoch], c=colours[epoch%2])
    objAx.plot(iterations, objLosses[epoch], c=colours[epoch%2])

# plt.ylim(0, 5000)
bboxAxBig.set_xticks(xMarkers)
bboxAxSmol.set_xticks(xMarkers)
clsAx.set_xticks(xMarkers)
objAx.set_xticks(xMarkers)
plt.savefig("./trainingGraph.png")