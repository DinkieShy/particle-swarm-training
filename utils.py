from sys import float_info

def bboxIOU(boxA, boxB, corners = False, widthHeight = False):
	assert corners or widthHeight, "Must specify bounding box format as [x1, y1, x2, y2] or [x, y, width, height]"

	if widthHeight:
		boxA = convertBbox(boxA)
		boxB = convertBbox(boxB)

	intersectWidth = max(boxA[2], boxB[2]) - min(boxA[0], boxB[0])
	intersectHeight = max(boxA[3], boxB[3]) - min(boxA[1], boxB[1])
	if intersectWidth <= 0 or intersectHeight <= 0:
		intersectArea = 0
	else:
		intersectArea = intersectWidth * intersectHeight

	boxAArea = (boxA[3] - boxA[1]) * (boxA[2] - boxA[1])
	boxBArea = (boxB[3] - boxB[1]) * (boxB[2] - boxB[1])
	unionArea = boxAArea + boxBArea - intersectArea

	iou = intersectArea / (unionArea + float_info.epsilon)
	return iou


def convertBbox(box):
	# converts from widthHeight to corner format
	x1 = box[0] - box[2]/2
	y1 = box[1] - box[3]/2
	x2 = box[0] + box[2]/2
	y2 = box[1] + box[3]/2

	return [x1, y1, x2, y2]