# Initially referenced from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/, plan to expand hence rewriting to understand fully
# YoloLoss from https://github.com/BobLiu20/YOLOv3_PyTorch/blob/master/nets/yolo_loss.py fork

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sys import float_info
import numpy as np
from utils import bboxIOU

from models import parseCfg, createModuleList

def predictTransform(prediction, inputDim, anchors, numClasses, CUDA=False):
	# TODO: Fix inputDim to reflect non-square images (in this function AND in YoloLoss.forward

	# This function takes the detection feature map from the last YOLO layer, and creates a tensor of the transformation between it and the predicted bounding box of the object

	batchSize = prediction.size(0)
	stride = inputDim // prediction.size(2)
	gridSize = inputDim // stride 
	bboxAttributes = 5 + numClasses
	numAnchors = len(anchors)

	# Anchors are defined in reference to input image size, rescale to fit prediction feature map
	anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

	prediction = prediction.view(batchSize, bboxAttributes*numAnchors, gridSize*gridSize)
	prediction = prediction.transpose(1, 2).contiguous()
	prediction = prediction.view(batchSize, gridSize*gridSize*numAnchors, bboxAttributes)

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
	def __init__(self, anchors, numClasses, imageSize):
		super(YoloLoss, self).__init__()
		self.anchors = anchors
		self.numAnchors = anchors # This is bad, please fix later <3
		self.numClasses = numClasses
		self.bboxAttributes = 5 + numClasses
		self.imageSize = imageSize

		self.ignoreThreshold = 0.5
		self.lambaXY = 2.5
		self.lambdaWH = 2.5
		self.lambdaConf = 1.0
		self.lambdaClass = 1.0

		self.mseLoss = nn.MSELoss()
		self.bceLoss = nn.BCELoss()

	def forward(self, batch, targets = None, CUDA=False):
		batchSize = batch.size(0)

		# \/ This is bad, please fix later <3 \/ #
		stride = self.imageSize[0] // batch.size(2)

		anchors = [(anchor[0]/stride, anchor[1]/stride) for anchor in anchors]

		prediction = batch.view(bs, self.numAnchors, self.bboxAttributes, inputDim, inputDim).permute(0, 1, 3, 4, 2).contiguous()

		x = torch.sigmoid(prediction[..., 0]) # Centres
		y = torch.sigmoid(prediction[..., 1])
		width = prediction[..., 2]
		height = prediction[..., 3]
		conf = torch.sigmoid(prediction[..., 4])
		classPred = torch.sigmoid(prediction[..., 5])

		if targets is not None:
			mask, invObjMask, tx, ty, tWidth, tHeight, tConf, tClassPred = self.getTarget(targets, anchors, inputDim)

			if CUDA:
				tx, ty, tWidth, tHeight, tConf, tClassPred = tx.cuda(), ty.cuda(), tWidth.cuda(), tHeight.cuda(), tConf.cuda(), tClassPred.cuda()

			lossX = self.bceLoss(x * mask, tx * mask)
			lossY = self.bceLoss(y * mask, ty * mask)
			lossWidth = self.mseLoss(w * mask, tw * mask)
			lossHeight = self.mseLoss(h * mask, th * mask)
			lossConf = self.bceLoss(conf * invObjMask, invObjMask * 0.0)
			lossClassPred = self.bceLoss(classPred[mask == 1], tClassPred[mask == 1])

			totalLoss = lossX*self.lambaXY + lossY * self.lambaXY + lossWidth * self.lambdaWH + lossHeight * self.lambdaWH + lossConf * self.lambdaConf + lossClassPred * self.lambdaClass

			return totalLoss, lossX.item(), lossY.item(), lossWidth.item(), lossHeight.item(), lossConf.item(), lossClassPred.item()
		else:
			if CUDA:
				FloatTensor = torch.cuda.FloatTensor
				LongTensor = torch.cuda.LongTensor
			else:
				FloatTensor = torch.FloatTensor
				LongTensor = torch.LongTensor
			gridX = torch.linspace(0, self.imageSize[0]-1, self.imageSize[0]).repeat(batchSize * self.numAnchors, 1, 1).view(x.shape).type(FloatTensor)
			gridY = torch.linspace(0, self.imageSize[1]-1, self.imageSize[1]).repeat(batchSize * self.numAnchors, 1, 1).view(y.shape).type(FloatTensor)
			anchorWidth = FloatTensor(anchors).index_select(1, torch.LongTensor([0])).repet(1, 1, self.imageSize[0]*self.imageSize[1]).view(width.shape)
			anchorHeight = FloatTensor(anchors).index_select(1, torch.LongTensor([1])).repeat(1, 1, self.imageSize[0]*self.imageSize[1]).view(height.shape)

			predictedBoxes = FloatTensor(prediction[...,:4].shape)
			predictedBoxes[...,0] = x.data + gridX
			predictedBoxes[...,1] = y.data + gridY
			predictedBoxes[...,2] = torch.exp(width.data) * anchorWidth
			predictedBoxes[...,3] = torch.exo(height.data) * anchorHeight

			scale = torch.Tensor([stride, stride] * 2).type(FloatTensor)
			output = torch.cat((predictedBoxes.view(batchSize, -1, 4) * scale, conf.view(batchSize, -1, 1), classPred.view(batchSize, -1, self.numClasses)), -1)

			return output.data

		def getTarget(self, targets, anchors, inputDim):
			batchSize = target.size(0)
			mask = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			invObjMask = torch.ones(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			tx = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			ty = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			tWidth = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			tHeight = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			tConf = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, requires_grad=False)
			tClassPred = torch.zeros(batchSize, self.numAnchors, inputDim, inputDim, self.numClasses, requires_grad=False)

			for item in range(batchSize):
				for t in range(target.shape[1]):
					if target[item, t].sum() == 0:
						continue

					gridX = target[item, t, 1] * inputDim
					gridY = target[item, t, 2] * inputDim
					gridWidth = target[item, t, 3] * inputDim
					gridHeight = target[item, t, 4] * inputDim

					gridIndexX = int(gridX)
					gridIndexY = int(gridY)

					gtBox = torch.FloatTensor(np.array([0, 0, gridWidth, gridHeight])).unsqueeze(0)
					anchorShapes = torch.FloatTensor(np.concatenate((np.zeros((self.numAnchors, 2)), np.array(anchors)), 1))

					anchorIOUs = bboxIOU(gtBox, anchorShapes)
					invObjMask[item, anchorIOUs > ignoreThreshold, gridIndexX, gridIndexY] = 0
					bestMatch = np.argmax(anchorIOUs)

					mask[item, bestMatch, gridIndexX, gridIndexY] = 1
					tx[item, bestMatch, gridIndexX, gridIndexY] = gridX - gridIndexX
					ty[item, bestMatch, gridIndexX, gridIndexY] = gridY - gridIndexY
					tWidth[item, bestMatch, gridIndexX, gridIndexY] = math.log(gridWidth/anchors[bestMatch][0] + float_info.epsilon)
					tHeight[item, bestMatch, gridIndexX, gridIndexY] = math.log(gridHeight/anchors[bestMatch][1] + float_info.epsilon)

					tConf[item, bestMatch, gridIndexX, gridIndexY] = 1
					tClassPred[item, bestMatch, gridIndexX, gridIndexY, int(target[item, t, 0])] = 1

			return mask, invObjMask, tx, ty, tWidth, tHeight, tConf, tClassPred
					
class Darknet(nn.Module):
	def __init__(self, cfgFile):
		super(Darknet, self).__init__()
		self.blocks = parseCfg(cfgFile)
		self.net_info, self.moduleList = createModuleList(self.blocks)

	def forward(self, x, CUDA=True):
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
				layers = [int(layer) for layers in module["layers"]]
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
				numClasses = int(module["classes"])

				x = x.data
				x = predictTransform(x, inputDim, anchors, numClasses, CUDA) 

				if not write:
					detections = x
					write = True
				else:
					detections = torch.cat((detections, x), 1)
		return detections
