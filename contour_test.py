import numpy as np
import cv2
image = cv2.imread('test.jpg')
#image = image[120:280, 210:330]
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

imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)
cv2.imshow("Gray", thresh)
cv2.waitKey()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output2 = output.copy()
#ahh = cv2.drawContours(output2, contours, -1, (0,255,0), 3)
c = max(contours, key=cv2.contourArea)
ahh = contours.index(c)
contours.pop(ahh)
#cv2.imshow("Gray", ahh)
#cv2.waitKey()

approx = cv2.approxPolyDP(c, 4, True)
print(approx)
# draw the approximated contour on the image
output3 = output.copy()
cv2.drawContours(output3, [approx], -1, (0, 255, 0), 3)

contours.pop(ahh)
c = max(contours, key=cv2.contourArea)
approx = cv2.approxPolyDP(c, 4, True)
cv2.drawContours(output3, [approx], -1, (0, 255, 0), 3)
print(approx)

# show the approximated contour image
cv2.imshow("Original v Approximated Contour", np.hstack([image,output,output3]))
cv2.waitKey(0)