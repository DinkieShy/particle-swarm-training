import sys
import json
import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("agg")

assert len(sys.argv) == 2, "Pass result filename"

wd = os.getcwd() + "/"
outputDir = wd + "swarmOutput/"
if not os.path.exists(outputDir):
    os.mkdir(outputDir)
else:
    for file in os.listdir(outputDir):
        os.remove(outputDir + file)

with open(wd + sys.argv[1]) as file:
    swarmOutput = json.load(file)
    file.close()

for ID in swarmOutput.keys():
    params = []
    results = []
    for point in range(len(swarmOutput[ID]["path"])):
        params.append([swarmOutput[ID]["path"][point][0][param] for param in swarmOutput[ID]["params"]])
        results.append(swarmOutput[ID]["path"][point][1])

    fig = plt.figure(figsize=(9.6,4.8))
    gridspec = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    ax0 = fig.add_subplot(gridspec[:,0], projection="3d")
    x = [i[0] for i in params]
    y = [i[1] for i in params]
    ax0.scatter(x, y, results)
    ax0.set_xlabel(swarmOutput[ID]["params"][0])
    ax0.set_ylabel(swarmOutput[ID]["params"][1])
    ax1 = fig.add_subplot(gridspec[0,1])
    ax1.scatter(x, results)
    ax1.set_xlabel(swarmOutput[ID]["params"][0])
    ax2 = fig.add_subplot(gridspec[1,1])
    ax2.scatter(y, results)
    ax2.set_xlabel(swarmOutput[ID]["params"][1])

    fig.tight_layout()
    fig.savefig(outputDir + ID + ".png")
    fig.clear()
    plt.close(fig)
