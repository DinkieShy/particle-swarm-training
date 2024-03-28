import numpy as np
import argparse
import os
import torch
from sys import float_info
import matplotlib
import matplotlib.pyplot as plt

def IOU(boxA, boxB):
    intersectWidth = min(boxA[0],boxB[0])
    intersectHeight = min(boxA[1],boxB[1])
    intersectArea = intersectHeight * intersectWidth
    unionArea = (boxA[0] * boxA[1]) + (boxB[0] * boxB[1]) - intersectArea
    return intersectArea / (unionArea + float_info.epsilon)

def euclideanDistance(pointA, pointB):
    pointA = np.array(pointA)
    pointB = np.array(pointB)
    return np.sqrt(np.sum(np.square(pointA-pointB)))

def initaliseClusters(annotations, k):
    # kmeans++, Arthur and Vassilvitskii (2007)
    initialSample = np.random.choice(annotations.shape[0])
    clusters = [annotations[initialSample]]
    for _ in range(k-1):
        distances = [-1 for _ in range(annotations.shape[0])]
        # distance from each point to it's nearest centroid (0 for points already selected)
        for i in range(annotations.shape[0]):
            clusterDistances = [euclideanDistance(annotations[i], cluster) for cluster in clusters]
            distances[i] = clusterDistances[np.argmin(clusterDistances)]
        clusters.append(annotations[np.argmax(distances)])
        # select the item with the highest distance as the next centroid

    return clusters

def main(k, annotations):
    done = False
    annotations = annotations.numpy()
    clusters = initaliseClusters(annotations, k)

    while not done:
        # Assign annotations to clusters
        newClusters = np.zeros((k, 2))
        newClustersAssignedCount = np.repeat(float_info.epsilon, k)
        totalIOU = 0
        for i in range(len(annotations)):
            IOUs = [IOU(cluster, annotations[i]) for cluster in clusters]
            assignedCluster = np.argmax(IOUs)
            totalIOU += IOUs[assignedCluster]
            newClusters[assignedCluster] += annotations[i]
            newClustersAssignedCount[assignedCluster] += 1

        newClustersAssignedCount = np.stack((newClustersAssignedCount, newClustersAssignedCount), axis=1)
        newClusters = np.round(newClusters/newClustersAssignedCount)

        if np.array_equal(newClusters, clusters):
            # nothing changed, done
            done = True
        else:
            # something changed, redo
            clusters = newClusters

        print(clusters)
        print(f"Avg IOU: {totalIOU/annotations.shape[0]}")

    return clusters, totalIOU/annotations.shape[0]

def experiment():
    parser = argparse.ArgumentParser(description='k means anchor selector')
    parser.add_argument('-k', type=int, default=None, metavar='k',
                        help='clusters to generate (default: None)')
    parser.add_argument("-d", '--image-dir', type=str, metavar="dir",
                        help="directory containing images and labels (default:None)")
    parser.add_argument("--minK", type=int, default=None, metavar="k",
                        help="min k value to use (default: None)")
    parser.add_argument("--maxK", type=int, default=None, metavar="k",
                        help="max k value to use (default: None)")
    args = parser.parse_args()
    assert args.image_dir is not None, "Directory not provided, use --image-dir or --help for more info"
    
    fileList = os.listdir(args.image_dir)
    if args.image_dir[-1] != "/":
        args.image_dir += "/"
    written = False
    for file in fileList:
        if (file[-3:].lower() == "png" or file[-3:].lower() == "jpg") and os.path.isfile(f"{args.image_dir}{file[:-4]}.csv"):
            # only run k-means on images with annotations, ignore non-annotated images and vice versa
            try:
                target = torch.load(f"{args.image_dir}{file[:-4]}.csv")
                if not written:
                    written = True
                    annotations = target["boxes"]
                else:
                    annotations = torch.cat((annotations, target["boxes"]), dim=0)
            except:
                assert False, "Not yet implemented for non-tensor annotations"
                # with open(f"{file[:-3]}.csv") as f:
                
    annotations = annotations[:,2:] - torch.stack((annotations[:,0], annotations[:,1]), dim=1)
    # Transforms tensor of bboxes to tensor of heights and widths (as if bboxes had been transformed to (0,0) and arbitrary 0's filtered out)
    print(f"Got {annotations.shape[0]} annotations")

    if args.k is not None:
        main(args.k, annotations)
    elif args.minK is not None and args.maxK is not None:
        results = {}
        for k in range(args.minK, args.maxK+1):
            _, IOU = main(k, annotations)
            print(f"\nAverage IOU for {k} clusters: {IOU}\n\n")
            results[k] = IOU

        print(results)

        # results = {1: 0.4224394329511745, 2: 0.5698596906817643, 3: 0.6459648562843469, 4: 0.6819995451880456, 5: 0.7096228456444768, 6: 0.7286906435837559, 7: 0.7435699139558364, 8: 0.7570097244507727, 9: 0.7642748900074671, 10: 0.772232698729697, 11: 0.7810083272509215, 12: 0.7885138104728358, 13: 0.7942127788012253, 14: 0.7978620632684187, 15: 0.8046862815130557}

        matplotlib.use("agg")
        plt.plot(list(results.keys()), results.values(), marker="^", markersize=7)
        plt.xticks(list(results.keys()))
        plt.title("IOU vs No. of Clusters")
        plt.xlim(1, args.maxK)
        plt.ylim(0.3, 0.83)
        plt.yticks(np.arange(0.4, 0.83, 0.05))
        plt.hlines(list(results.values()), 0, list(results.keys()), linestyles="dotted")
        plt.ylabel("Average IOU")
        plt.xlabel("K value")
        plt.savefig("./kmeansResults.png")
    else:
        assert False, "Must provide either [-k] or [--maxK and --minK]"

if __name__ == "__main__":
    experiment()