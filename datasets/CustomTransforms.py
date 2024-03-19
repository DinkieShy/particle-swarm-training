import math
import numpy as np
from PIL import Image

def resize(image, boxes, newSize):
	resizedImage = image.resize(newSize)
	widthScale = resizedImage.size[0] / image.size[0]
	heightScale = resizedImage.size[1] / image.size[1]
	boxes[:,0] *= widthScale
	boxes[:,1] *= heightScale
	boxes[:,2] *= widthScale
	boxes[:,3] *= heightScale

	return resizedImage, boxes


def shear(image, boxes, amount, useXAxis = True, returnMatrix = False):
	shearAmount = np.random.rand()*amount
	transformMatrix = np.identity(3)

	if useXAxis:
		transformMatrix[0][1] = shearAmount
	else:
		transformMatrix[1][0] = shearAmount

	if returnMatrix:
		return transformMatrix
	else:
		image, boxes = applyAffineTransform(image, boxes, transformMatrix)
	return image, boxes

def rotate(image, boxes, radians, returnMatrix = False):
	rotateAmount = np.random.rand()*radians
	clockwise = np.random.rand() > 0.5
	
	if clockwise:
		rotateAmount *= -1

	transformMatrix = np.identity(3)
	cosTheta = np.cos(rotateAmount)
	sinTheta = np.sin(rotateAmount)
	transformMatrix[0][0] = cosTheta
	transformMatrix[0][1] = (-1) * sinTheta
	transformMatrix[1][0] = sinTheta
	transformMatrix[1][1] = cosTheta

	if returnMatrix:
		return transformMatrix
	else:
		image, boxes = applyAffineTransform(image, boxes, transformMatrix)
	return image, boxes

def shearAndRotate(image, boxes, shearAmount, useXAxis, rotateAmount):
	rotateMatrix = rotate(image, boxes, rotateAmount, returnMatrix=True)
	shearMatrix = shear(image, boxes, shearAmount, useXAxis=useXAxis, returnMatrix=True)
	transformMatrix = rotateMatrix.dot(shearMatrix)

	image, boxes = applyAffineTransform(image, boxes, transformMatrix)
	return image, boxes

def gaussianNoise(image, boxes, mean=1., scale=0.125):
	width, height = image.size
	newImage = np.array(image)
	noise = np.random.normal(mean, scale, size=(height, width, 3))
	newImage = image*noise
	image = Image.fromarray(newImage.astype(np.uint8), 'RGB')

	return image, boxes

def applyAffineTransform(image, boxes, matrix):
	inv = np.linalg.inv(matrix)

	width, height = image.size

	topLeft = matrix.dot(np.array([0, 0, 1]))
	topRight = matrix.dot(np.array([width, 0, 1]))
	bottomLeft = matrix.dot(np.array([0, height, 1]))
	bottomRight = matrix.dot(np.array([width, height, 1]))

	minX = min(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0])
	maxX = max(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0])
	minY = min(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1])
	maxY = max(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1])

	newWidth = math.ceil(maxX-minX)
	newHeight = math.ceil(maxY-minY)

	if minX < 0 or minY < 0:
		matrix[0][2] = -minX if minX < 0 else 0
		matrix[1][2] = -minY if minY < 0 else 0
		inv = np.linalg.inv(matrix)

	pilTransform = (inv[0][0], inv[0][1], inv[0][2],
	inv[1][0], inv[1][1], inv[1][2])

	image = image.transform((newWidth, newHeight), Image.AFFINE, data = pilTransform, resample = Image.BICUBIC)

	transformedBoxes = []
	for box in boxes:
		topLeft = matrix.dot(np.array([box[0], box[1], 1]))
		topRight = matrix.dot(np.array([box[2], box[1], 1]))
		bottomLeft = matrix.dot(np.array([box[0], box[3], 1]))
		bottomRight = matrix.dot(np.array([box[2], box[3], 1]))

		minX = int(min(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]))
		maxX = int(max(topLeft[0], topRight[0], bottomLeft[0], bottomRight[0]))
		minY = int(min(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]))
		maxY = int(max(topLeft[1], topRight[1], bottomLeft[1], bottomRight[1]))

		transformedBoxes.append([minX, minY, maxX, maxY])

	return image, transformedBoxes