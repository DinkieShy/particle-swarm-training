# Initially referenced from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/, plan to expand hence rewriting to understand fully

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sys import float_info
import numpy as np
from utils import bboxIOU, convertBbox
from itertools import chain

from models import parseCfg, createModuleList, YoloDetection

def predictTransform(prediction, inputDim, anchors, numClasses, CUDA=False):
	# TODO: Fix inputDim to reflect non-square images (in this function AND in YoloLoss.forward

	# This function takes the detection feature map from the last YOLO layer, and creates a tensor of the transformation between it and the predicted bounding box of the object

	if not CUDA:
		device = "cpu"
	else:
		device = "cuda"

	# print(prediction.shape)
	batchSize = prediction.size(0)
	strideW = inputDim[0] // prediction.size(2)
	strideH = inputDim[1] // prediction.size(2)
	gridSize = (inputDim[0] // strideW, inputDim[1] // strideH)
	bboxAttributes = 5 + numClasses
	numAnchors = len(anchors)

	# print(batchSize, bboxAttributes*numAnchors, gridSize[0]*gridSize[1])
	prediction = prediction.view(batchSize, bboxAttributes*numAnchors, gridSize[0]*gridSize[1])
	prediction = prediction.transpose(1, 2).contiguous()
	prediction = prediction.view(batchSize, gridSize[0]*gridSize[1]*numAnchors, bboxAttributes)

	# Anchors are defined in reference to input image size, rescale to fit prediction feature map
	anchors = [(anchor[0]/strideW, anchor[1]/strideH) for anchor in anchors]

	# Apply sigmoid to centreX, centreY and objectness score
	prediction[:,:,0] = prediction[:,:,0].sigmoid()
	prediction[:,:,1] = prediction[:,:,1].sigmoid()
	# prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

	a, b = np.meshgrid(np.arange(gridSize[0]), np.arange(gridSize[1])) # Makes a list of coordinate matricies

	# Tensor.view shares data with underlying tensor, in this case it applies the prediction to the grid we just defined
	xOffset = torch.as_tensor(a, device=device).view(-1, 1)
	yOffset = torch.as_tensor(b, device=device).view(-1, 1)

	xyOffset = torch.cat((xOffset, yOffset), 1).repeat(1, numAnchors).view(-1,2).unsqueeze(0)

	prediction[:,:,:2] += xyOffset

	anchors = torch.tensor(anchors, dtype=torch.float32, device=device)

	anchors = anchors.repeat(gridSize[0]*gridSize[1], 1).unsqueeze(0)
	prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

	# prediction[:,:,5:5 + numClasses] = torch.sigmoid(prediction[:,:,5:5 + numClasses])

	# resize to image 
	prediction[:,:,0] *= strideW
	prediction[:,:,2] *= strideW
	prediction[:,:,1] *= strideH
	prediction[:,:,3] *= strideH

	return prediction

def computeLoss(outputs, targets, model):
	# output is in the format [yolo  layer][batchSize, anchor, x, y, objectness, class logits...]
	device = outputs[0].device
	clsLoss, bboxLoss, objLoss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
	bboxLossAvgCount = 0

	clsTarget, bboxTarget, anchors = buildTargets(outputs, targets, model, device=device)
	clsLossFunc = nn.BCEWithLogitsLoss()
	objLossFunc = nn.BCEWithLogitsLoss()

	for yoloLayer in range(len(outputs)):
		for batch in range(clsTarget[yoloLayer].shape[0]):
			numTargets = clsTarget[yoloLayer].shape[1]
			gridTargets = []
			targetAnchors = [model.anchorGroups[yoloLayer][int(anchors[yoloLayer][batch, i])] for i in range(numTargets)]
			for i in range(numTargets):
				gridTargets.append([int(torch.floor(i)) for i in bboxTarget[yoloLayer][batch,i,:2]])
			# print(gridTargets) # <- Base cls and box loss on these indicies
			objTarget = torch.zeros_like(outputs[yoloLayer][...,0], device=device)

			for target in range(numTargets):
				targetX, targetY = min(gridTargets[i][0], outputs[yoloLayer].shape[2]-1), min(gridTargets[i][1], outputs[yoloLayer].shape[3]-1)
				anchor = int(anchors[yoloLayer][batch,target])

				for xCoord in range(outputs[yoloLayer].shape[2]):
					for yCoord in range(outputs[yoloLayer].shape[3]):
						boxPrediction = torch.zeros(4, device=device)
						boxPrediction[0] = torch.sigmoid(outputs[yoloLayer][batch,anchor,xCoord,yCoord,0]) + xCoord
						boxPrediction[1] = torch.sigmoid(outputs[yoloLayer][batch,anchor,xCoord,yCoord,1]) + yCoord
						boxPrediction[2] = torch.exp(outputs[yoloLayer][batch,anchor,xCoord,yCoord,2])*targetAnchors[target][0]
						boxPrediction[3] = torch.exp(outputs[yoloLayer][batch,anchor,xCoord,yCoord,3])*targetAnchors[target][1]

						iou = bboxIOU(boxPrediction, bboxTarget[yoloLayer][batch,target], widthHeight=True)
						if xCoord == targetX and yCoord == targetY:
							objTarget[batch,anchor,xCoord,yCoord] = iou
							clsLoss += clsLossFunc(outputs[yoloLayer][batch,anchor,xCoord,yCoord,5:], clsTarget[yoloLayer][batch,target])
							bboxLoss += 1.0 - iou
							bboxLossAvgCount += 1
						elif iou < 0.5:
							bboxLoss += 1.0 - iou
							bboxLossAvgCount += 1

			objLoss += objLossFunc(outputs[yoloLayer][...,4], objTarget)

	bboxLoss /= bboxLossAvgCount
	bboxLoss *= 0.05
	clsLoss *= 0.5

	loss = bboxLoss + clsLoss + objLoss
	print(loss.item())

	return loss, (bboxLoss, clsLoss, objLoss)

def buildTargets(outputs, targets, model, device="cpu"):
	# Need to convert targets to YOLO coord space and expand class logits tensor

	batchSize = len(targets)
	imageSize = (int(model.net_info["width"]), int(model.net_info["height"]))
	targetTensors = []
	for i in range(batchSize):
		targetTensors.append(torch.empty((1, 0)))
		for box in range(len(targets[i]["boxes"])):
			newTarget = torch.zeros(5, device=device) # want tensor in form [x, y, width, height, class]
			w = targets[i]["boxes"][box][2] - targets[i]["boxes"][box][0]
			h = targets[i]["boxes"][box][3] - targets[i]["boxes"][box][1]
			x = targets[i]["boxes"][box][0] + w/2
			y = targets[i]["boxes"][box][3] + h/2
			cls = targets[i]["labels"][box] - 1
			newTarget[0], newTarget[1], newTarget[2], newTarget[3], newTarget[4] = x, y, w, h, cls
			if targetTensors[i].shape[1] == 0:
				targetTensors[i] = newTarget.unsqueeze(0)
			else:
				targetTensors[i] = torch.cat((targetTensors[i], newTarget.unsqueeze(0)))

	clsTarget, bboxTarget, targetAnchors = [], [], []


	yoloLayers = [i for i in range(len(model.moduleList)) if model.blocks[i]["type"] == "yolo"]
	for i, layer in enumerate(yoloLayers):
		clsTarget.append(torch.zeros(1, device=device))
		bboxTarget.append(torch.zeros(1, device=device))
		targetAnchors.append(torch.zeros(1, device=device))
		write = False
		for batch in range(batchSize):
			numTargets = len(targetTensors[batch])
			targets = targetTensors[batch]

			# Stack targets w.r.t anchors to make output size
			anchorGroupSize = len(model.anchorGroups[0])
			anchorIndices = torch.arange(anchorGroupSize, device=device).float().view(anchorGroupSize, 1).repeat(1, numTargets)
			targets = torch.cat((targets.repeat(anchorGroupSize, 1, 1), anchorIndices[:,:,None]), 2)
			
			# Move targets and anchors into YOLO coord space
			stride = tuple([imageSize[ii]/outputs[i].shape[2+ii] for ii in range(2)])
			targets[...,0] /= stride[0]
			targets[...,2] /= stride[0]
			targets[...,1] /= stride[1]
			targets[...,3] /= stride[1]

			anchors = torch.Tensor([[anchor[0]/stride[0], anchor[1]/stride[1]] for anchor in model.moduleList[layer][0].anchors])
			numAnchors = len(anchors)
			# Need to find best matching anchor, but can't use IoU because the anchors get scaled; check ratio of heights/widths instead
			ratios = targets[...,2:3] / anchors[:,None]
			bestMatches = torch.zeros((numAnchors, numTargets), device=device).bool()
			# ratios are in format [anchor index, target]
			for ii in range(numTargets):
				bestScore = -1
				for iii in range(ratios.shape[0]):
					score = compareRatio(ratios[iii, ii, 0], ratios[iii, ii, 1])
					if score < bestScore or bestScore == -1:
						bestScore = score
						bestMatch = iii
				bestMatches[bestMatch, ii] = True
			# targets is now in the form [x, y, width, height, class, anchor]

			clsLogits = torch.zeros((numTargets, int(model.net_info["numClasses"])), device=device)
			targetAnchorsTensor = torch.zeros_like(clsLogits[...,0])
			for ii in range(numTargets):
				clsLogits[ii, int(targets[bestMatches][ii, 4])] = 1
				targetAnchorsTensor[ii] = targets[bestMatches][ii, 5]

			if not write:
				clsTarget[-1] = clsLogits.unsqueeze(0)
				bboxTarget[-1] = targets[bestMatches][...,:4].unsqueeze(0)
				targetAnchors[-1] = targetAnchorsTensor.unsqueeze(0)
				write = True
			else:
				clsTarget[-1] = torch.cat((clsTarget[-1], clsLogits.unsqueeze(0)))
				bboxTarget[-1] = torch.cat((bboxTarget[-1], targets[bestMatches][...,:4].unsqueeze(0)))
				targetAnchors[-1] = torch.cat((targetAnchors[-1], targetAnchorsTensor.unsqueeze(0)))

	return clsTarget, bboxTarget, targetAnchors

def compareRatio(width, height):
	# returns a score of how similar the boxes are, ignoring scale
	score = abs(1-width) + abs(1-height)
	invScore = abs(1-(1/width)) + abs(1-(1/height))
	return min(score, invScore)
					
class Darknet(nn.Module):
	def __init__(self, cfgFile):
		super(Darknet, self).__init__()
		self.blocks = parseCfg(cfgFile)
		self.net_info, self.moduleList = createModuleList(self.blocks)
		self.losses = []
		self.anchorGroups = []
		self.lossFuncs = {}
		self.iterDone = False
		self.layersToStore = []
		self.grid = torch.zeros(1)
		self.yoloOutputs = []

	def makeGrid(self, xSize, ySize, device="cpu"):
		y, x = torch.meshgrid([torch.arange(ySize), torch.arange(xSize)], indexing="ij")
		return torch.stack((x, y), 2).view((1, 1, ySize, xSize, 2)).float().to(device)

	def forward(self, x, target=None, CUDA=True):
		outputs = {} # Store feature maps for route layers later
		self.yoloOutputs = []
		self.anchorGroups = []
		write = False # Flag used to track when to initalise tensor of detection feature maps
		if CUDA:
			device = "cuda"
		else:
			device = "cpu"
		for index, module in enumerate(self.blocks):
			moduleType = module["type"]
			if moduleType == "net":
				continue

			if moduleType == "convolutional" or moduleType == "upsample":
				x = self.moduleList[index](x)
				if (not self.iterDone) or(self.iterDone and index in self.layersToStore):
					outputs[index] = x

			elif moduleType == "route":
				layers = [int(layer) for layer in module["layers"].split(",")]
				if layers[0] > 0:
					layers[0] -= index

				if len(layers) == 1:
					x = outputs[index + (layers[0])]
					if (not self.iterDone):
						self.layersToStore.append(index+layers[0])
				else:
					if layers[1] > 0:
						layers[1] -= index
					if (not self.iterDone):
						self.layersToStore.append(index+layers[0])
						self.layersToStore.append(index+layers[1])
					# Dear me:	if you're looking at an error where dimensions don't match up,
					# 			check the input size is divisible by 32
					featureMaps = (outputs[index + layers[0]], outputs[index + layers[1]])
					x = torch.cat(featureMaps, 1)
				if (not self.iterDone) or(self.iterDone and index in self.layersToStore):
					outputs[index] = x

			elif moduleType == "shortcut":
				layer = int(module["from"])
				x = outputs[index-1] + outputs[index+layer]
				if (not self.iterDone) or (self.iterDone and index in self.layersToStore):
					outputs[index] = x

				if (not self.iterDone):
					self.layersToStore.append(index-1)
					self.layersToStore.append(index+layer)

			elif moduleType == "yolo":
				# print(x.shape)
				# x = self.moduleList[index][0](x)
				imageSize = (int(self.net_info["width"]), int(self.net_info["height"]))
				anchors = self.moduleList[index][0].anchors
				self.anchorGroups.append(anchors)
				bboxAttributes = 5 + int(self.net_info["numClasses"])

				batchSize, _, xSize, ySize = x.shape
				x = x.view(batchSize, len(anchors), bboxAttributes, xSize, ySize).permute(0, 1, 3, 4, 2).contiguous()
				# Gets network output in the format [batchSize, anchor, xCoord, yCoord, objectness, class logits...]

				if not self.training:
					# Do inference, convert from YOLO coordinate space to image
					stride = tuple([imageSize[i]/x.size(2+i) for i in range(2)])
					if self.grid.shape[2:4] != x.shape[2:4]:
						self.grid = self.makeGrid(xSize, ySize, device=device)
					# Add grid to convert from offset to coords, multiply by stride to get coords in image space
					x[..., 0] = (x[...,0] + self.grid[...,0]) * stride[0]
					x[..., 1] = (x[...,1] + self.grid[...,1]) * stride[1]
					# Multiply width/height by anchors
					x[..., 2:4] = torch.exp(x[...,2:4]) * torch.tensor(list(chain(*anchors))).view(1, -1, 1, 1, 2)
					# Sigmoid logits
					x[..., 4:] = x[..., 4:].sigmoid()
					x = x.view(batchSize, -1, bboxAttributes)

				self.yoloOutputs.append(x)

		self.iterDone = True # Only store layer outputs we NEED from now on
		return self.yoloOutputs
