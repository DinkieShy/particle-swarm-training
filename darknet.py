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
	if CUDA:
		xOffset = torch.FloatTensor(a).view(-1, 1).cuda()
		yOffset = torch.FloatTensor(b).view(-1, 1).cuda()
	else:
		xOffset = torch.FloatTensor(a).view(-1, 1)
		yOffset = torch.FloatTensor(b).view(-1, 1)

	xyOffset = torch.cat((xOffset, yOffset), 1).repeat(1, numAnchors).view(-1,2).unsqueeze(0)

	prediction[:,:,:2] += xyOffset

	if CUDA:
		anchors = torch.FloatTensor(anchors).cuda()
	else:
		anchors = torch.FloatTensor(anchors)

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

		self.mseLoss = nn.MSELoss()
		self.bceLoss = nn.BCELoss()

	def forward(self, output, targets, CUDA=False):
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

		# Anchor sizes are determined via k-means clustering- need to do this on our dataset?

		batchSize = output.size(0)
		strideWidth = self.imageSize[0]/self.mapSize[0]
		strideHeight = self.imageSize[1]/self.mapSize[1]
		output = output.view(batchSize, self.numAnchors, self.mapSize[0], self.mapSize[1], self.bboxAttributes)

		x = output[..., 0]
		y = output[..., 1]
		width = output[..., 2]
		height = output[..., 3]
		conf = output[..., 4]
		classPred = output[..., 5:]

		targetX = torch.zeros(x.shape)
		targetY = torch.zeros(x.shape)
		targetWidth = torch.zeros(x.shape)
		targetHeight = torch.zeros(x.shape)
		targetConf = torch.zeros(x.shape)
		targetClassPred  = torch.zeros(classPred.shape)

		# print(output.shape)
		# Output shape is [batchSize, number of anchors, feature map width, feature map height, 5 + numClasses]
		# There are predictions for every pixel for every anchor size

		for index in range(batchSize):
			imageTargetBoxes = targets["boxes"][index]
			imageTargetClasses = torch.zeros(self.numClasses)
			imageTargetClasses[targets["labels"][index][0]] = 1
			for i in range(1, len(imageTargetBoxes)):
				boxClass = torch.zeros(self.numClasses)
				boxClass[targets["labels"][index][i]-1] = 1
				imageTargetClasses = torch.stack(((imageTargetClasses), boxClass))

			for i in range(len(imageTargetBoxes)):
				x1, y1, x2, y2 = imageTargetBoxes[i]
				imageTargetBoxes[i][0] = (x1+x2)/2
				imageTargetBoxes[i][1] = (y1+y2)/2		
				imageTargetBoxes[i][2] = x2 - x1
				imageTargetBoxes[i][3] = y2 - y1	

			imageTargets = torch.cat((imageTargetBoxes, imageTargetClasses), dim=1)
			if CUDA:
				imageTargets = imageTargets.cuda()

			# ^ creates tensor of [[xCenter, yCenter, width, height, classBool, classBool, ... , classBool]]

			mask = torch.zeros((batchSize, self.numAnchors, self.mapSize[0], self.mapSize[1]))
			# >0 : pixel overlaps GT box with stored index
			# -1 : pixel does not correspond to a GT box
			exactMatches = torch.zeros(mask.shape)

			for i in range(len(imageTargets)):
				# Select anchor with closest area
				bestDiff = -1
				bestMatch = -1
				for anchor in range(len(self.anchors)):
					diff = abs(self.anchorAreas[anchor] - targets["area"][index][i])
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
						x[index,anchor,xCoord,yCoord] -= xCoord
						y[index,anchor,xCoord,yCoord] -= yCoord

						if exactMatches[index, anchor, xCoord, yCoord] == 1:
							# exact match 
							targetMatched = imageTargets[int(mask[index, anchor, xCoord, yCoord])]
							targetX[index,anchor,xCoord,yCoord] = targetMatched[0]-xCoord
							targetY[index,anchor,xCoord,yCoord] = targetMatched[1]-yCoord
							targetWidth[index,anchor,xCoord,yCoord] = targetMatched[2]
							targetHeight[index,anchor,xCoord,yCoord] = targetMatched[3]
							targetConf[index,anchor,xCoord,yCoord] = 1
							targetClassPred[index,anchor,xCoord,yCoord] = targetMatched[5:]
						else:
							box = output[index,anchor,xCoord,yCoord,:]
							bboxIOUs = [bboxIOU(convertBbox(box), targetBox, corners=True) for targetBox in targets["boxes"][index]]
							bestFitTarget = bboxIOUs.index(max(bboxIOUs))

							# Not assigned to GT, ignore prediction (except objectness)
							x[index,anchor,xCoord,yCoord] *= 0
							y[index,anchor,xCoord,yCoord] *= 0
							width[index,anchor,xCoord,yCoord] *= 0
							height[index,anchor,xCoord,yCoord] *= 0
							classPred[index,anchor,xCoord,yCoord] *= torch.zeros(self.numClasses)

							if bboxIOUs[bestFitTarget] > 0.5:
								# overlaps with GT box, also ignore objectness
								conf[index,anchor,xCoord,yCoord] *= 0
							
		# print(targetClassPred[0,0])
		# input(classPred)
			
		xLoss = self.mseLoss(x, targetX)
		yLoss = self.mseLoss(y, targetY)
		widthLoss = self.mseLoss(width, targetWidth)
		heightLoss = self.mseLoss(height, targetHeight)
		confLoss = self.bceLoss(conf, targetConf)
		classPredLoss = self.bceLoss(classPred, targetClassPred)

		loss =  (xLoss + yLoss) * self.lambaXY + \
				(widthLoss + heightLoss) * self.lambdaWH + \
				confLoss * self.lambdaConf + \
				classPredLoss * self.lambdaClass

		return loss, xLoss.item(), yLoss.item(), widthLoss.item(), heightLoss.item(), confLoss.item(), classPredLoss.item()

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

	def forward(self, x, target=None, CUDA=True):
		outputs = {} # Store feature maps for route layers later
		write = False # Flag used to track when to initalise tensor of detection feature maps
		for index, module in enumerate(self.blocks):
			moduleType = module["type"]
			if moduleType == "net":
				continue

			if moduleType == "convolutional" or moduleType == "upsample":
				x = self.moduleList[index](x)
				outputs[index] = x

			elif moduleType == "route":
				layers = [int(layer) for layer in module["layers"].split(",")]
				if layers[0] > 0:
					layers[0] -= index

				if len(layers) == 1:
					x = outputs[index + (layers[0])]
				else:
					if layers[1] > 0:
						layers[1] -= index
					featureMaps = (outputs[index + layers[0]], outputs[index + layers[1]])
					x = torch.cat(featureMaps, 1)
				outputs[index] = x

			elif moduleType == "shortcut":
				layer = int(module["from"])
				x = outputs[index-1] + outputs[index+layer]
				outputs[index] = x

			elif moduleType == "yolo":
				anchors = self.moduleList[index][0].anchors
				inputDim = int(self.net_info["height"])
				numClasses = int(self.net_info["numClasses"])

				x = x.data
				x = predictTransform(x, inputDim, anchors, numClasses, CUDA) 

				if not write:
					self.detections = x
					write = True
				else:
					self.detections = torch.cat((self.detections, x), 1)

		return self.detections
