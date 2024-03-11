# Initially referenced from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/, plan to expand hence rewriting to understand fully

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
	# Empty layer, used for transferring feature maps from earlier layers in YOLO
	def __init__(self):
		super(EmptyLayer, self).__init__()

class YoloDetection(nn.Module):
	def __init__(self, anchors):
		super(YoloDetection, self).__init__()
		self.anchors = anchors

def parseCfg(filepath):
	with open(filepath) as file:
		lines = [line.rstrip().lstrip() for line in file.readlines() if len(line) > 0]
		file.close()

	newBlock = {}
	blocks = []

	for line in lines:
		if len(line) == 0 or line[0] == "#":
			continue
		if line[0] == "[":
			if len(newBlock) > 0:
				blocks.append(newBlock)
				newBlock = {}
			newBlock["type"] = line[1:-1]
		else:
			key, value = line.split("=")
			newBlock[key.rstrip()] = value.lstrip()
	blocks.append(newBlock)
	return blocks

def createModuleList(blocks):
	netInfo = blocks.pop(0) # first block is always net info
	moduleList = nn.ModuleList()
	prevChannels = 3 # source calls these 'filters', channels makes more sense to me. Either way it's the depth of the feature map (so 3 for RGB images)
	outputChannels = []

	for index, block in enumerate(blocks):
		newModule = nn.Sequential()
		match block["type"]:
			case "convolutional":
				batchNormalize = "batch_normalize" in block
				bias = not batchNormalize
				channels = int(block["filters"])
				kernelSize = int(block["size"])
				stride = int(block["stride"])

				if "padding" in block:
					pad = kernelSize - 1
				else:
					pad = 0

				conv = nn.Conv2d(prevChannels, channels, kernelSize, stride, pad, bias=bias)
				newModule.add_module(f"conv_{index}", conv)
				if batchNormalize:
					bn = nn.BatchNorm2d(channels)
					newModule.add_module(f"batch_norm_{index}", bn)
				match block["activation"]:
					case "leaky":
						activation = nn.LeakyReLU(0.1, inplace=True)
					case "mish":
						activation == nn.Mish(inplace=True)
				newModule.add_module(f"leaky_{index}", activation)

			case "upsample":
				stride = int(block["stride"])
				upsample = nn.Upsample(scale_factor=2, mode="bilinear")
				newModule.add_module(f"upsample_{index}", upsample)

			case "route":
				# Use feature map from previous layer
				# The *actual* concatenation is done in the darknet forward pass, it really doesn't need it's own layer
				targets = [int(layer) for layer in block["layers"].split(",")]
				start = targets[0]
				if len(targets) > 1:
					end = targets[1]
				else:
					end = 0

				if start > 0:
					start -= index
				if end > 0:
					end -= index

				route = EmptyLayer()
				newModule.add_module(f"route_{index}", route)

				if end < 0:
					channels = outputChannels[index + start] + outputChannels[index + end]
				else:
					channels = outputChannels[index + start]

			case "shortcut":
				# Skip connection
				shortcut = EmptyLayer()
				newModule.add_module(f"shortcut_{index}", shortcut)

			case "yolo":
				masks = [int(mask) for mask in block["mask"].split(",")]
				anchors = [int(anchor) for anchor in block["anchors"].split(",")]
				anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
				anchors = [anchors[i] for i in masks]

				detection = YoloDetection(anchors)
				newModule.add_module(f"detection_{index}", detection)

		moduleList.append(newModule)
		prevChannels = channels
		outputChannels.append(channels)

	return (netInfo, moduleList)

def main():
	blocks = parseCfg("./cfg/yolov3.cfg")
	print(createModuleList(blocks))

if __name__ == "__main__":
	main()