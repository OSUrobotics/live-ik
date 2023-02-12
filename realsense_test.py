## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

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
        #if not first:
        #    cv2.imwrite("test_red_second.jpg", image)
        #else:
        #    first = False


        # Remove background - Set pixels further than clipping_distance to black
        grey_color = 0#153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
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

        # Generate mask from binary slices of individual color channels
        #([60, 15, 120], [120, 80, 200])
        # This is BGR
        # 0 is blue
        # 1 is red
        # 2 is green
        #print(color_image)

        boundaries = [
                #([120, 120, 200], [160, 170, 255]), # orange pla
                #([50, 30, 120], [80, 70, 180])  # red paint
                #([60, 60, 15], [140, 140, 70]) # green
                #([130, 85, 10], [160, 105, 70]) # blue
                #([70, 35, 120], [130, 80, 200])
                ([25, 15, 60], [120, 70, 160])
            ]

        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")
            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(bg_removed, lower, upper)
            output = cv2.bitwise_and(bg_removed, bg_removed, mask = mask)
            # show the images
            #cv2.imshow("hey", np.hstack([image, output]))
            
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        
        imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 50, 255, 0)
        ##cv2.imshow("Gray", thresh)
        #cv2.waitKey(0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        output2 = output.copy()
        #ahh = cv2.drawContours(output2, contours, -1, (0,255,0), 3)
        #ahh = np.argmax(contours, key=cv2.contourArea)
        if contours:
            #print(contours)
            c = max(contours, key=cv2.contourArea)
            #ahh = contours.index(c)
            #contours.pop(ahh)
            #cv2.imshow("Gray", ahh)
            #cv2.waitKey()

            approx = cv2.approxPolyDP(c, 6, True)
            # draw the approximated contour on the image
            
            output3 = output.copy()
            cv2.drawContours(output3, [approx], -1, (0, 255, 0), 3)

            # Print the location of the approximated contour

            print(vtx[approx[0][0][1]][approx[0][0][0]], vtx[approx[1][0][1]][approx[1][0][0]])
            
            
            cv2.namedWindow("Original v Approximated Contour")
            cv2.setMouseCallback("Original v Approximated Contour",mouseRGB)
            # show the approximated contour image
            #print()
            cv2.imshow("Original v Approximated Contour", np.hstack([image,output,output3]))
            cv2.waitKey(0)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        #color_image = np.array(color_image, dtype = "uint8")
        #filtered = (color_image[:,:,0] > 25) & (color_image[:,:,0] < 105) & (color_image[:,:,1] > 10) & (color_image[:,:,1] < 85) & (color_image[:,:,2] > 85) & (color_image[:,:,2] < 150)# & (color_image[:,:,1] < 80) & (color_image[:,:,0] > 120) & (color_image[:,:,0] < 200)
        
        
        
        
        
        '''

        # Slice Color Image with mask
        color_image[:, :, 0] = color_image[:, :, 0] * filtered
        color_image[:, :, 1] = color_image[:, :, 1] * filtered
        color_image[:, :, 2] = color_image[:, :, 2] * filtered

        out = cv2.bitwise_and(color_image, color_image)
        imgray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 10, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hh = cv2.drawContours(out, contours, -1, (0,255,0), 3)
        cv2.imshow("show",np.hstack((out, hh)))
        #images = np.hstack((imgray)) #depth_colormap))
        cv2.namedWindow('hey')
        cv2.setMouseCallback('hey',mouseRGB)
        cv2.namedWindow('hey')
        cv2.imshow('hey', thresh)
        key = cv2.waitKey(1)

        
        out = cv2.bitwise_and(color_image, color_image)

        
        imgray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(imgray, 50, 255, 0)
        cv2.imshow("Gray", out)
        cv2.waitKey()
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output2 = out.copy()
        hh = cv2.drawContours(output2, contours, -1, (0,255,0), 3)
        cv2.imshow("show",np.hstack((out, hh)))
        cv2.waitKey(0)

        
        ahh = cv2.drawContours(output2, contours, -1, (0,255,0), 3)
#

        #ahh = cv2.drawContours(output2, contours, -1, (0,255,0), 3)
        c = max(contours, key=cv2.contourArea)
        print(c)
        ahh = contours.index(c)
        contours.pop(ahh)
        #cv2.imshow("Gray", ahh)
        #cv2.waitKey()

        approx = cv2.approxPolyDP(c, 4, True)
        print(approx)
        # draw the approximated contour on the image
        output3 = color_image.copy()
        cv2.drawContours(output3, [approx], -1, (0, 255, 0), 3)

        contours.pop(ahh)
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 4, True)
        cv2.drawContours(output3, [approx], -1, (0, 255, 0), 3)
        print(approx)

        # show the approximated contour image
        cv2.imshow("Original v Approximated Contour", np.hstack([image,color_image,output3]))
        key = cv2.waitKey(1)

        '''
        '''
        # Slice Depth Image with mask
        depth_image = depth_image * filtered
        #depth_image = depth_image * filtered
        #depth_image = depth_image * filtered

        # Slice pointcloud with mask 
        vtx = vtx[filtered > 0] # [x,][0] = (x_1, y_1, z_1)

        # Covert pointcloud from array of tupples to full numpy array [3, x]
        vtxStack = np.stack([vtx['f0'],vtx['f1'],vtx['f2']]) 

        # print(np.shape(depth_image), np.shape(vtx))
        # print(np.shape(vtx[filtered > 0]))

        # Calculate average point from all remaining points
        vtxAverage = np.mean(vtxStack, axis = 1)

        # Print for diagnostics
        #print(vtxAverage)

        # Render images:
        #   depth align to color on left
        #   depth on right
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #images = np.hstack((imgray)) #depth_colormap))
        cv2.namedWindow('hey')
        cv2.setMouseCallback('hey',mouseRGB)
        cv2.namedWindow('hey')
        cv2.imshow('hey', imgray)
        key = cv2.waitKey(1)
        '''
        # Press esc or 'q' to close the image window
        #if key & 0xFF == ord('q') or key == 27:
        #    cv2.destroyAllWindows()
        #    break
        
        
finally:
    pipeline.stop()