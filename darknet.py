# Initially referenced from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/, plan to expand hence rewriting to understand fully

import torch
import torch.nn as nn
import numpy as np
from itertools import chain

from models import parseCfg, createModuleList

def computeIOUs(output, targetBox):
	device = output.device
	desiredShape = list(output.shape)
	desiredShape[-1] = 8
	boxPredictions = torch.zeros(desiredShape, device=device)
	boxPredictions[...,:4] = output[...,:4]
	boxPredictions[...,:2] += torch.floor(targetBox[:2])
	boxPredictions[...,4:6] = output[...,0:2] - output[...,2:4]/2
	boxPredictions[...,6:8] = output[...,0:2] + output[...,2:4]/2
	targetBoxCorners = torch.zeros_like(targetBox)
	targetBoxCorners[:2] = targetBox[2:] - targetBox[:2]/2
	targetBoxCorners[2:] = targetBox[2:] + targetBox[:2]/2
	desiredShape[-1] = 2
	intersects = torch.zeros(desiredShape, device=device)
	intersects[...,0] = torch.minimum(boxPredictions[...,6], targetBoxCorners[2])-torch.maximum(boxPredictions[...,4],targetBoxCorners[0])
	intersects[...,1] = torch.minimum(boxPredictions[...,7], targetBoxCorners[3])-torch.maximum(boxPredictions[...,5],targetBoxCorners[1])
	intersects = torch.prod(torch.clamp(intersects,min=0), dim=-1)
	ious = intersects / ((torch.prod(boxPredictions[...,2:4], dim=-1) + torch.prod(targetBox[2:4], dim=-1)) - intersects)

	return ious

def computeLoss(outputs, targets, model):
	# torch.autograd.set_detect_anomaly(True)
	# output is in the format [yolo  layer][batchSize, anchor, x, y, objectness, class logits...]
	device = outputs[0].device
	clsLoss, bboxLoss, objLoss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
	bboxLossAvgCount = 0

	# Convert targets to coord space of YOLO layer
	clsTarget, bboxTarget, anchors = buildTargets(outputs, targets, model, device=device)
	clsLossFunc = nn.BCEWithLogitsLoss()
	objLossFunc = nn.BCEWithLogitsLoss()
	bboxLossFunc = nn.MSELoss()

	for yoloLayer in range(len(outputs)):
		outputs[yoloLayer][...,0:2] = outputs[yoloLayer][...,0:2].sigmoid()
		for anchor in range(len(model.anchorGroups[yoloLayer])):
			outputs[yoloLayer][:,anchor,...,2:4] = outputs[yoloLayer][:,anchor,...,2:4].exp()*torch.tensor(model.anchorGroups[yoloLayer][anchor], device=device)
		mask = torch.zeros_like(outputs[yoloLayer][...,0], device=device) # Mask of which cells incur Objectness loss
		objTarget = torch.zeros_like(outputs[yoloLayer][...,4], device=device) # Targets for objectness score, should be 0 for any cells not containing a target
		for batch in range(len(clsTarget[yoloLayer])):
			numTargets = clsTarget[yoloLayer][batch].shape[0]
			gridTargets = [] # Grid cells which contain the center of a target bbox
			for i in range(numTargets):
				gridTargets.append([int(torch.floor(i)) for i in bboxTarget[yoloLayer][batch][i,:2]])

			for target in range(numTargets):
				anchor = int(anchors[yoloLayer][batch][target]) # Only incur loss for anchor with target
				mask[batch,anchor,...] = 1 # Mask of which cells incur Objectness loss

			for target in range(numTargets):
				targetX, targetY = min(gridTargets[i][0], outputs[yoloLayer].shape[2]-1), min(gridTargets[i][1], outputs[yoloLayer].shape[3]-1)
				# Restrict to valid coords in case target center outside coord space somehow
				anchor = int(anchors[yoloLayer][batch][target]) # Only incur loss for anchor with target

				ious = computeIOUs(outputs[yoloLayer][batch,anchor,...], bboxTarget[yoloLayer][batch][target])
				mask[batch,anchor,ious > 0.5] = 0 # Ignore cells with IoU > 0.5

				objTarget[batch,anchor,targetX,targetY] = 1 # Cell should have predicted a target
				clsLoss += clsLossFunc(outputs[yoloLayer][batch,anchor,targetX,targetY,5:], clsTarget[yoloLayer][batch][target])
				bboxLoss += bboxLossFunc(outputs[yoloLayer][batch,anchor,targetX,targetY,:4], (bboxTarget[yoloLayer][batch][target] - torch.tensor([targetX,targetY,0,0],device=device)))
				bboxLossAvgCount += 1
			for target in range(numTargets):
				targetX, targetY = min(gridTargets[i][0], outputs[yoloLayer].shape[2]-1), min(gridTargets[i][1], outputs[yoloLayer].shape[3]-1)
				anchor = int(anchors[yoloLayer][batch][target])
				mask[batch,anchor,targetX,targetY] = 1 # Need to incur loss for target cell regardless of IoU

			# mask[batch,anchor,...] is still 0 if anchor was not used to detect a target, no loss is incurred
			# additionally, mask[batch,anchor,x,y] is 0 if it's prediction overlapped with a GT by IoU 0.5 despite it not being the center
		objLoss += objLossFunc(outputs[yoloLayer][...,4]*mask, objTarget)

	bboxLoss /= bboxLossAvgCount
	bboxLoss *= 0.05
	clsLoss *= 0.5

	loss = bboxLoss + clsLoss + objLoss

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
			y = targets[i]["boxes"][box][1] + h/2
			cls = targets[i]["labels"][box] - 1
			newTarget[0], newTarget[1], newTarget[2], newTarget[3], newTarget[4] = x, y, w, h, cls
			if targetTensors[i].shape[1] == 0:
				targetTensors[i] = newTarget.unsqueeze(0)
			else:
				targetTensors[i] = torch.cat((targetTensors[i], newTarget.unsqueeze(0)))

	clsTarget, bboxTarget, targetAnchors = [], [], []


	yoloLayers = [i for i in range(len(model.moduleList)) if model.blocks[i]["type"] == "yolo"]
	for i, layer in enumerate(yoloLayers):
		clsTarget.append([])
		bboxTarget.append([])
		targetAnchors.append([])
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

			anchors = torch.tensor([[anchor[0]/stride[0], anchor[1]/stride[1]] for anchor in model.moduleList[layer][0].anchors], device=device)
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

			clsTarget[-1].append(clsLogits)
			bboxTarget[-1].append(targets[bestMatches][...,:4])
			targetAnchors[-1].append(targetAnchorsTensor)

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
					# Dear programmer:	if you're looking at an error where dimensions don't match up,
					# 					check the input size is divisible by 32
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
					x[..., 2:4] = torch.exp(x[...,2:4]) * torch.tensor(list(chain(*anchors)),device=device).view(1, -1, 1, 1, 2)
					# Sigmoid logits
					x[..., 4:] = x[..., 4:].sigmoid()
					x = x.view(batchSize, -1, bboxAttributes)

				self.yoloOutputs.append(x)

		self.iterDone = True # Only store layer outputs we NEED from now on
		return self.yoloOutputs
