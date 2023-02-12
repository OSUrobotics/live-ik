import cv2
import numpy as np
import math
import imutils
 

# Read an image, a window and bind the function to window
image = cv2.imread("test_red.jpg")
image = image[120:280, 210:330]
 

boundaries = [
	#([120, 120, 200], [160, 170, 255]), # orange pla
    #([50, 30, 120], [80, 70, 180])  # red paint
    #([120, 130, 65], [160, 180, 112]) # green
    #([130, 85, 10], [160, 105, 70]) # blue
    #([70, 35, 120], [130, 80, 200])
    ([60, 15, 120], [120, 80, 200])
]

for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    # show the images
    #cv2.imshow("hey", np.hstack([image, output]))
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

'''
image = cv2.imread("test_red.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the contour and center of the shape on the image
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
	cv2.putText(image, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
'''
gray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey()
edges = cv2.Canny(gray,50,200,None, 3)
cv2.imshow("Edges", edges)
cv2.waitKey()


# Copy edges to the images that will display the results in BGR
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 20, None, 0, 0)

if lines is not None:
    print('here')
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)


linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    print('here')
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

cv2.imshow("Source", output)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

cv2.waitKey()




'''
cv2.imshow("hey", edges)
cv2.waitKey(0)



# convert to gray
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

# threshold the grayscale image
ret, thresh = cv2.threshold(gray,0,255,0)

# find outer contour
cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

# get rotated rectangle from outer contour
rotrect = cv2.minAreaRect(cntrs[0])
box = cv2.boxPoints(rotrect)
box = np.int0(box)

# draw rotated rectangle on copy of img as result
result = output.copy()
cv2.drawContours(result,[box],0,(0,0,255),2)
angle = rotrect[-1]
cv2.imshow("THRESH", thresh)
cv2.imshow("RESULT", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

"""
#print(edges)
lines = cv2.HoughLines(edges,1,np.pi/180,50)
print(lines)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
cv2.destroyAllWindows()
"""