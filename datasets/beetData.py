import torch
from torchvision.transforms.functional import to_tensor as convertToTensor
from torchvision.transforms import ToPILImage as convertToImageTransform
from torchvision.utils import save_image as saveImageTensor
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import math

from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import datasets.CustomTransforms as CustomTransforms

class BeetDataset(Dataset):
	def __init__(self, image_list):
		super().__init__()

		self.images = []
		self.labels = []
		self.training = False

		# assert os.path.isdir(image_list[0]), "image directory not found"

		for filename in image_list:
			if (filename[-3:].lower() == "jpg" or filename[-3:].lower() == "png") and os.path.isfile(filename[:-3] + "txt"):
				self.images.append(filename)
				self.labels.append(filename[:-3] + "txt")

	def __getitem__(self, index: int):
		assert index < len(self.images), "index error when accessing dataset"

		image_id = self.images[index]

		imagePath = self.images[index]
		labelsPath = self.labels[index]

		image = Image.open(imagePath)#.convert("RGB")
		labelsFile = open(labelsPath)
		lines = labelsFile.readlines()
		labelsFile.close()

		boxes = []
		labels = []
		areas = []
		width, height = image.size

		for line in lines:
			splitLine = line.split(" ")
			labels.append(int(splitLine[0])+1) # 0 is reserved for background class
			box = [float(splitLine[1])*width, float(splitLine[2])*height, float(splitLine[3])*width, float(splitLine[4])*height]
			box[0] -= box[2]/2
			box[1] -= box[3]/2
			box[2] += box[0]
			box[3] += box[1]
			boxes.append(box)

		if width > height:
			image, boxes = CustomTransforms.rotate(image, boxes, -math.pi/2)
		image, boxes = CustomTransforms.resize(image, boxes, (416, 416))

		# displayImage = ImageDraw.Draw(image)
		# for box in boxes:
		# 	displayImage.rectangle(box, fill=None, outline="red")
		# image.show()
		# input()

		for box in boxes:
			area = (box[2]-box[0])*(box[3]-box[1])
			areas.append(area)

		if self.training:
			randomNumber = np.random.rand()
			if randomNumber <= 0.5: # Gaussian noise
				image, boxes = CustomTransforms.gaussianNoise(image, boxes)
			randomNumber = np.random.rand()
			if randomNumber <= 0.3: # Shear or rotate
				randomNumber = np.random.rand()
				if randomNumber <= (1./3.): # Just rotate
					image, boxes = CustomTransforms.rotate(image, boxes, math.pi/6)
				elif randomNumber <= (2./3.): # Just shear
					image, boxes = CustomTransforms.shear(image, boxes, 0.125, useXAxis = np.random.rand(1) > 0.5)
				else: # both
					image, boxes = CustomTransforms.shearAndRotate(image, boxes, 0.125, np.random.rand(1) > 0.5, math.pi/6)

		labels = torch.as_tensor(labels, dtype=torch.int64)
		areas = torch.as_tensor(areas, dtype=torch.float32)
		iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
		boxes = torch.as_tensor(boxes, dtype=torch.float32)

		# image = F.normalize(transforms.ToTensor()(image))
		image = transforms.ToTensor()(image)


		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["image_id"] = torch.tensor([index])
		target["area"] = areas
		target["iscrowd"] = iscrowd

		return image, target

	def __len__(self) -> int:
		return len(self.images)

class AugmentedBeetDataset(Dataset):
	def __init__(self, imageListFile, transform = None):
		# Expects a path to a .txt containing filepaths to images
		super().__init__()

		self.images = []
		self.training = False
		self.transform = transform

		# assert os.path.isdir(imageList[0]), "image directory not found"
		assert os.path.exists(imageListFile), "image list path not found"

		imageList = readImageListFile(imageListFile)

		for filename in imageList:
			if (filename[-3:].lower() == "jpg" or filename[-3:].lower() == "png") and (os.path.isfile(filename[:-3] + "csv")):
				self.images.append(filename.rstrip().lstrip())

	def __getitem__(self, index: int):
		assert index < len(self.images), "index error when accessing dataset"

		imagePath = self.images[index]
		image = Image.open(imagePath)#.convert("RGB")
		targets = torch.load(imagePath[:-3] + "csv")

		if self.transform is not None:
			image, targets = self.transform(image, targets)

		return image, targets

	def __len__(self) -> int:
		return len(self.images)

def readImageListFile(path):
	file = open(path)
	imageList = file.readlines()
	for i in range(len(imageList)):
		imageList[i] = imageList[i].replace("\n", "")
		if imageList[i][0] != "." and imageList[i][0] != "/":
			imageList[i] = "/" + imageList[i]
	file.close()
	return imageList

def main():

	toImage = convertToImageTransform()

	if not os.path.exists("/home/grey/datasets/LincolnAugment416/"):
		os.mkdir("/home/grey/datasets/LincolnAugment416/")

	if os.path.exists("/home/grey/datasets/LincolnAugment416/all"):
		listDir = os.listdir("/home/grey/datasets/LincolnAugment416/all")
		for i in listDir:
			os.remove(os.path.join("/home/grey/datasets/LincolnAugment416/all", i))
		os.rmdir("/home/grey/datasets/LincolnAugment416/all")
	os.mkdir("/home/grey/datasets/LincolnAugment416/all")

	trainDS = BeetDataset(readImageListFile("/datasets/sugar_beet_all/train.txt"))
	trainDS.training = True
	valDS = BeetDataset(readImageListFile("/datasets/sugar_beet_all/val.txt"))
	testDS = BeetDataset(readImageListFile("/datasets/sugar_beet_all/test.txt"))

	subSets = [trainDS, valDS, testDS]
	subsetFiles = ["/datasets/LincolnAugment416/train.txt", "/datasets/LincolnAugment416/val.txt", "/datasets/LincolnAugment416/test.txt"]
	for i in subsetFiles:
		file = open(i, "w")
		file.close()

	for ii in range(len(subSets)):
		for i in range(len(subSets[ii])):
			imageFiles = []
			imageTensor, baseTargets = subSets[ii][i]
			baseImage = toImage(imageTensor)
			imageFilename = subSets[ii].images[i].split("/")[-1]
			imageFiles.append(f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-base.{imageFilename[-3:]}")
			saveImageTensor(imageTensor, imageFiles[-1])
			torch.save(baseTargets, f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-base.csv")

			if subSets[ii].training:
				gaussImage, _ = CustomTransforms.gaussianNoise(baseImage, baseTargets["boxes"])
				imageFiles.append(f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-gauss.{imageFilename[-3:]}")
				gaussImage.save(imageFiles[-1])
				torch.save(baseTargets, f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-gauss.csv")

				rotateImage, rotateBoxes = CustomTransforms.rotate(baseImage, baseTargets["boxes"], math.pi/6)
				rotateTargets = baseTargets.copy()
				rotateTargets["boxes"] = torch.as_tensor(rotateBoxes, dtype=torch.float32)
				imageFiles.append(f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-rotate.{imageFilename[-3:]}")
				rotateImage.save(imageFiles[-1])
				torch.save(rotateTargets, f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-rotate.csv")

				shearImage, shearBoxes = CustomTransforms.shear(baseImage, baseTargets["boxes"], 0.125, useXAxis = np.random.rand(1) > 0.5)
				shearTargets = baseTargets.copy()
				shearTargets["boxes"] = torch.as_tensor(shearBoxes, dtype=torch.float32)
				imageFiles.append(f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-shear.{imageFilename[-3:]}")
				shearImage.save(imageFiles[-1])
				torch.save(shearTargets, f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-shear.csv")

				shearAndRotateImage, shearAndRotateBoxes = CustomTransforms.shearAndRotate(baseImage, baseTargets["boxes"], 0.125, np.random.rand(1) > 0.5, math.pi/6)
				shearAndRotateTargets = baseTargets.copy()
				shearAndRotateTargets["boxes"] = torch.as_tensor(shearAndRotateBoxes, dtype=torch.float32)
				imageFiles.append(f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-shearAndRotate.{imageFilename[-3:]}")
				shearAndRotateImage.save(imageFiles[-1])
				torch.save(shearAndRotateTargets, f"/datasets/LincolnAugment416/all/{imageFilename[:-4]}-shearAndRotate.csv")

			with open(subsetFiles[ii], "a") as file:
				for i in imageFiles:
					file.write(i + "\n")

	return

if __name__=="__main__":
	main()