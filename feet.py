# USAGE
# python object_size.py --image images/example_01.png --width 0.955

#two parameters: image and width of left-most object


import argparse
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import cv
import cv2

def midpoint(A, B):
	return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Input Image Path")
ap.add_argument("-w", "--width", type=float, required=True,
	help="Left most object's width (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

contour = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
contour = imutils.grab_contours(contour)

(contour, _) = contours.sort_contours(contour)
pixelsPerMetric = None


for c in contour:

	if cv2.contourArea(c) < 100:
		continue

	orig = image.copy()
	hull = cv2.convexHull(c)
	box = cv2.minAreaRect(hull)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	box = perspective.order_points(box) # Reordering the bounding box
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)


	(tl, tr, br, bl) = box
	(tophorizx, tophorizy) = midpoint(tl, tr)
	(bottomhorizx, bottomhorizy) = midpoint(bl, br)

	(leftvertx, leftverty) = midpoint(tl, bl)
	(rightvertx, rightverty) = midpoint(tr, br)
	cv2.circle(orig, (int(tophorizx), int(tophorizy)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(bottomhorizx), int(bottomhorizy)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(leftvertx), int(leftverty)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(rightvertx), int(rightverty)), 5, (255, 0, 0), -1)
	cv2.line(orig, (int(tophorizx), int(tophorizy)), (int(bottomhorizx), int(bottomhorizy)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(leftvertx), int(leftverty)), (int(rightvertx), int(rightverty)),
		(255, 0, 255), 2)

	dA = dist.euclidean((tophorizx, tophorizy), (bottomhorizx, bottomhorizy))
	dB = dist.euclidean((leftvertx, leftverty), (rightvertx, rightverty))

	#Only if it none, because we don't want to recalculate pixelsperMetric after first 
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	cv2.putText(orig, "{:.1f}in".format(dimA),
		(int(tophorizx - 15), int(tophorizy - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(rightvertx + 10), int(rightverty)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	cv2.imshow("Image", orig)
	cv2.waitKey(0)