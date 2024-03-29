# import the necessary packages

import imutils
import cv2
import sys

ARUCO_DICT = {
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
}

# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread("Images/test_red_second.jpg")
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, aruco_dict, parameters=arucoParams)

print(corners) 
if not corners:
	print("hiiii")
frame_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
cv2.imshow("ho", frame_markers)
cv2.waitKey(0)