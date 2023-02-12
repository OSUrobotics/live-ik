# import the necessary packages
import numpy as np
import cv2

# load the image
image = cv2.imread("test_red.jpg")

# define the list of boundaries
## Remeber BGR not RGB!!
boundaries = [
	#([120, 120, 200], [160, 170, 255]), # orange pla
    ([70, 35, 120], [130, 80, 200])  # red paint
    #([120, 130, 65], [160, 180, 112]) # green
    #([130, 85, 10], [160, 105, 70]) # blue
]

def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = image[y,x,0]
        colorsG = image[y,x,1]
        colorsR = image[y,x,2]
        colors = image[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    # show the images
    cv2.namedWindow('hey')
    cv2.setMouseCallback('hey',mouseRGB)
    cv2.imshow("hey", np.hstack([image, output]))
    cv2.namedWindow('hey')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


   #https://stackoverflow.com/questions/58632469/how-to-find-the-orientation-of-an-object-shape-python-opencv 