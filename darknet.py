# Initially referenced from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/, plan to expand hence rewriting to understand fully

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sys import float_info
import numpy as np
from utils import bboxIOU, convertBbox

from models import parseCfg, createModuleList

def predictTransform(prediction, inputDim, anchors, numClasses, CUDA=False):
	# TODO: Fix inputDim to reflect non-square images (in this function AND in YoloLoss.forward

	# This function takes the detection feature map from the last YOLO layer, and creates a tensor of the transformation between it and the predicted bounding box of the object

	if not CUDA:
		device = "cpu"
	else:
		device = "cuda"

	batchSize = prediction.size(0)
	stride = inputDim // prediction.size(2)
	gridSize = inputDim // stride 
	bboxAttributes = 5 + numClasses
	numAnchors = len(anchors)

	prediction = prediction.view(batchSize, bboxAttributes*numAnchors, gridSize*gridSize)
	prediction = prediction.transpose(1, 2).contiguous()
	prediction = prediction.view(batchSize, gridSize*gridSize*numAnchors, bboxAttributes)

	# Anchors are defined in reference to input image size, rescale to fit prediction feature map
	anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

	# Apply sigmoid to centreX, centreY and objectness score
	prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
	prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
	prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

	grid = np.arange(gridSize)
	a, b = np.meshgrid(grid, grid) # Makes a list of coordinate matricies

	# Tensor.view shares data with underlying tensor, in this case it applies the prediction to the grid we just defined
	xOffset = torch.Tensor(a, dtype=torch.float32, device=device).view(-1, 1)
	yOffset = torch.Tensor(b, dtype=torch.float32, device=device).view(-1, 1)

	xyOffset = torch.cat((xOffset, yOffset), 1).repeat(1, numAnchors).view(-1,2).unsqueeze(0)

	prediction[:,:,:2] += xyOffset

	anchors = torch.Tensor(anchors, dtype=torch.float32, device=device)

	anchors = anchors.repeat(gridSize**2, 1).unsqueeze(0)
	prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

	prediction[:,:,5:5 + numClasses] = torch.sigmoid(prediction[:,:,5:5 + numClasses])

	# resize to image 
	prediction[:,:,:4] *= stride

	return prediction

class YoloLoss(nn.Module):
	def __init__(self, anchors, numClasses, mapSize, imageSize):
		super(YoloLoss, self).__init__()
		self.anchors = anchors
		self.anchorAreas = [a[0] * a[1] for a in self.anchors]
		self.numAnchors = len(anchors)
		self.numClasses = numClasses
		self.bboxAttributes = 5 + numClasses
		self.mapSize = mapSize
		self.imageSize = (int(imageSize[0]), int(imageSize[1]))

		self.nmsConf = 0
		self.confThreshold = 0
		self.lambaXY = 2.5
		self.lambdaWH = 2.5
		self.lambdaConf = 1.0
		self.lambdaClass = 1.0

		self.mseLoss = nn.MSELoss(reduction="sum")
		self.bceLoss = nn.BCELoss(reduction="sum")

	def forward(self, output, targets):
		"""
		Helpful extract from yolov3 paper

		YOLOv3 predicts an objectness score for each bounding
		box using logistic regression. This should be 1 if the bound-
		ing box prior overlaps a ground truth object by more than
		any other bounding box prior. If the bounding box prior
		is not the best but does overlap a ground truth object by
		more than some threshold we ignore the prediction, follow-
		ing [17]. We use the threshold of .5. Unlike [17] our system
		only assigns one bounding box prior for each ground truth
		object. If a bounding box prior is not assigned to a ground
		truth object it incurs no loss for coordinate or class predic-
		tions, only objectness
		"""

		device = output.get_device()
		if device == -1:
			device = "cpu"
		else:
			device = "cuda"

		output = torch.nan_to_num(output)

		# Anchor sizes are determined via k-means clustering- need to do this on our dataset?

		batchSize = output.size(0)
		strideWidth = self.imageSize[0]/self.mapSize[0]
		strideHeight = self.imageSize[1]/self.mapSize[1]
		output = output.view(batchSize, self.numAnchors, self.mapSize[0], self.mapSize[1], self.bboxAttributes)

		# x = output[..., 0]
		# y = output[..., 1]
		# width = output[..., 2]
		# height = output[..., 3]
		# conf = output[..., 4]
		# classPred = output[..., 5:]

		# targetX = torch.zeros(output[..., 0].shape, device=device)
		# targetY = torch.zeros(targetX.shape, device=device)
		# targetWidth = torch.zeros(targetX.shape, device=device)
		# targetHeight = torch.zeros(targetX.shape, device=device)
		# targetConf = torch.zeros(targetX.shape, device=device)
		# targetClassPred  = torch.zeros(output[..., 5:].shape, device=device)

		bboxTarget = torch.zeros(output[...,:4].shape, device=device)
		objectClassTarget = torch.zeros(output[...,4:].shape, device=device)

		# print(output.shape)
		# Output shape is [batchSize, number of anchors, feature map width, feature map height, 5 + numClasses]
		# There are predictions for every pixel for every anchor size

		for index in range(batchSize):
			imageTargetBoxes = targets[index]["boxes"]
			imageTargetClasses = torch.zeros((1,self.numClasses), device=device)
			imageTargetClasses[0,targets[index]["labels"][0]-1] = 1
			for i in range(1, len(imageTargetBoxes)):
				boxClass = torch.zeros(self.numClasses, device=device)
				boxClass[targets[index]["labels"][i]-1] = 1
				imageTargetClasses = torch.cat(((imageTargetClasses), boxClass.unsqueeze(0)), dim=0)

			for i in range(len(imageTargetBoxes)):
				x1, y1, x2, y2 = imageTargetBoxes[i]
				imageTargetBoxes[i][0] = (x1+x2)/2
				imageTargetBoxes[i][1] = (y1+y2)/2		
				imageTargetBoxes[i][2] = x2 - x1
				imageTargetBoxes[i][3] = y2 - y1	

			imageTargets = torch.cat((imageTargetBoxes, imageTargetClasses), dim=1)
			# ^ creates tensor of [[xCenter, yCenter, width, height, classBool, classBool, ... , classBool]]

			mask = torch.zeros((batchSize, self.numAnchors, self.mapSize[0], self.mapSize[1]), device=device)
			# >0 : pixel overlaps GT box with stored index
			# -1 : pixel does not correspond to a GT box
			exactMatches = torch.zeros(mask.shape, device=device)

			for i in range(len(imageTargets)):
				# Select anchor with closest area
				bestDiff = -1
				bestMatch = -1
				for anchor in range(len(self.anchors)):
					diff = abs(self.anchorAreas[anchor] - targets[index]["area"][i])
					if diff < bestDiff or bestDiff == -1:
						bestDiff = diff
						bestMatch = anchor

				targetCenter = [int((imageTargets[i][2]-imageTargets[i][0])/strideWidth), int((imageTargets[i][3]-imageTargets[i][1])/strideHeight)]

				mask[index,bestMatch,targetCenter[0],targetCenter[1]] = i
				exactMatches[index, bestMatch, targetCenter[0], targetCenter[1]] = 1
				# mask[index, anchor, xCenter, yCenter] = 1 if *that* pixel corresponds to a gt box
				# Only this pixel incurs x/y/class loss
				# For other pixels, no loss incurred if 0.5 IoU with a gt box, else only objectness loss

			for xCoord in range(self.mapSize[0]):
				for yCoord in range(self.mapSize[1]):
					for anchor in range(len(self.anchors)):
						output[index,anchor,xCoord,yCoord,0] -= xCoord
						output[index,anchor,xCoord,yCoord,1] -= yCoord

						if exactMatches[index, anchor, xCoord, yCoord] == 1:
							# exact match 
							targetMatched = imageTargets[int(mask[index, anchor, xCoord, yCoord])]
							bboxTarget[index,anchor,xCoord,yCoord,0] = targetMatched[0]-xCoord
							bboxTarget[index,anchor,xCoord,yCoord,1] = targetMatched[1]-yCoord
							bboxTarget[index,anchor,xCoord,yCoord,2] = targetMatched[2]
							bboxTarget[index,anchor,xCoord,yCoord,3] = targetMatched[3]
							objectClassTarget[index,anchor,xCoord,yCoord,0] = 1
							objectClassTarget[index,anchor,xCoord,yCoord,1:] = targetMatched[5:]
						else:
							box = output[index,anchor,xCoord,yCoord,:4]
							bboxIOUs = [bboxIOU(convertBbox(box), targetBox, corners=True) for targetBox in targets[index]["boxes"]]
							bestFitTarget = bboxIOUs.index(max(bboxIOUs))

							# Not assigned to GT, ignore prediction
							conf = output[index,anchor,xCoord,yCoord,4]
							output[index,anchor,xCoord,yCoord] *= 0

							if bboxIOUs[bestFitTarget] < 0.5:
								# doesn't overlap with GT box, don't ignore objectness
								output[index,anchor,xCoord,yCoord,4] = conf
			
		bboxPred = output[...,:4]
		confClassPred = output[...,4:]
		bboxLoss = self.mseLoss(bboxPred, bboxTarget)
		confLoss = self.bceLoss(confClassPred, objectClassTarget)

		loss = bboxLoss + confLoss

		return loss

def filterUnique(tensor):
	npTensor = tensor.cpu().numpy()
	uniqueNP = np.unique(npTensor)
	uniqueTensor = torch.from_numpy(uniqueNP)

	newTensor = tensor.new(uniqueTensor.shape)
	newTensor.copy_(uniqueTensor)
	return newTensor
					
class Darknet(nn.Module):
	def __init__(self, cfgFile):
		super(Darknet, self).__init__()
		self.blocks = parseCfg(cfgFile)
		self.net_info, self.moduleList = createModuleList(self.blocks)
		self.losses = []
		self.lossFuncs = {}
		self.iterDone = False
		self.layersToStore = []

	def forward(self, x, target=None, CUDA=True):
		outputs = {} # Store feature maps for route layers later
		write = False # Flag used to track when to initalise tensor of detection feature maps
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
				anchors = self.moduleList[index][0].anchors
				inputDim = int(self.net_info["height"])
				numClasses = int(self.net_info["numClasses"])
				mapSize = x.shape

				x = predictTransform(x, inputDim, anchors, numClasses, CUDA) 

				if self.training:
					if mapSize[2] not in self.lossFuncs.keys():
						self.lossFuncs[mapSize[2]] = YoloLoss(anchors, numClasses, (mapSize[2], mapSize[3]), (self.net_info["width"], self.net_info["height"]))

				if not write:
					self.detections = x
					write = True
					if self.training:
						loss = self.lossFuncs[mapSize[2]](x, target)
						self.losses.append(loss)
				else:
					self.detections = torch.cat((self.detections, x), 1)
					if self.training:
						newLoss = self.lossFuncs[mapSize[2]](x, target)
						self.losses.append(newLoss)

		self.iterDone = True # Only store layer outputs we NEED from now on
		if self.training:
			return self.detections, self.losses
		return self.detections
