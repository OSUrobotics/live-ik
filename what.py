# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
from matplotlib import pyplot as plt

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = .38 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Create pointcloud object
pc = rs.pointcloud()

first = True

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Get pointcloud from depth image
        points = pc.calculate(aligned_depth_frame)

        # Get individual points from pointcloud

        vtx = np.asanyarray(points.get_vertices()).reshape(480, 640)
        #print("shape")
        #print(np.shape(vtx))
        #print(vtx)

        # Extract Color and Depth Image
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        image = color_image.copy()


        # Remove background - Set pixels further than clipping_distance to black
        grey_color = 0#153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = color_image.copy()#np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        #color_image = bg_removed
        

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

        lowcolor = (0,0,90)
        highcolor = (70,70,200)
        thresh = cv2.inRange(bg_removed, lowcolor, highcolor)

        # apply morphology close
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get contours and filter on area
        #result = color_image.copy()
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sort_cont = sorted(contours, key=cv2.contourArea, reverse=True)
        c = sort_cont[0]#max(contours, key=cv2.contourArea)
            #ahh = contours.index(c)
            #contours.pop(ahh)
            #cv2.imshow("Gray", ahh)
            #cv2.waitKey()
            # Was 6
        #approx = cv2.approxPolyDP(c, 3, True)
        #print("contour")
        #print(contours)
        #print("c")
        #print(c)

        epsilon = 0.1*cv2.arcLength(c,True)
        approx1 = cv2.approxPolyDP(c,epsilon,True)
            # draw the approximated contour on the image
            
        output3 = color_image.copy()
        #cv2.drawContours(output3, [approx], -1, (0, 255, 0), 3)

        #contours = contours[0] if len(contours) == 2 else contours[1]
        #for c in contours:
        #    area = cv2.contourArea(c)
        #    if area > 5000:
        #        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)

        
        c = sort_cont[1]
        epsilon = 0.05*cv2.arcLength(c,True)
        approx2 = cv2.approxPolyDP(c,epsilon,True)
            # draw the approximated contour on the image
        

        print(approx1)
        print(approx2)
            
        output3 = color_image.copy()
        cv2.drawContours(output3, [approx1, approx2], -1, (0, 255, 0), 3)

          
        cv2.namedWindow("Original v Approximated Contour")
        cv2.setMouseCallback("Original v Approximated Contour",mouseRGB)

        cv2.imshow("Original v Approximated Contour", output3)
        cv2.waitKey(1)

        
finally:
    pipeline.stop()

import cv2
import numpy as np

# read image as grayscale
img = cv2.imread('red_line.png')

# threshold on red color
lowcolor = (0,0,75)
highcolor = (50,50,135)
thresh = cv2.inRange(img, lowcolor, highcolor)


# apply morphology close
kernel = np.ones((5,5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# get contours and filter on area
result = img.copy()
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
result = img.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 5000:
        cv2.drawContours(result, [c], -1, (0, 255, 0), 2)


# show thresh and result    
cv2.imshow("thresh", thresh)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save resulting images
cv2.imwrite('red_line_thresh.png',thresh)
cv2.imwrite('red_line_extracted.png',result)