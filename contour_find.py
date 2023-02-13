import cv2
import numpy as np
import sys


class ContourFind:
    def __init__(self) -> None:

        # Arrays for masking the image (BGR format)
        self.lowcolor = (0,0,90)
        self.highcolor = (70,70,200)
    
        pass

    def find_countours(self, color_image):

        thresh = cv2.inRange(color_image, self.lowcolor, self.highcolor)

        # apply morphology close
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # get contours and filter on area
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours largest to smallest by area
        sort_cont = sorted(contours, key=cv2.contourArea, reverse=True)

        # Grab the largest contour and apprximate it to the two endpoints
        c = sort_cont[0]
        epsilon = 0.1*cv2.arcLength(c,True)
        approx_cont_1 = cv2.approxPolyDP(c,epsilon,True)

        # Grab the second largest contour and apprximate it to the two endpoints
        c = sort_cont[1]
        epsilon = 0.05*cv2.arcLength(c,True)
        approx_cont_2 = cv2.approxPolyDP(c,epsilon,True)

        # Get in the format we want
        try:
            cont_1 = np.array([approx_cont_1[0][0], approx_cont_1[1][0]])
            cont_2 = np.array([approx_cont_2[0][0], approx_cont_2[1][0]])

            # Check that we are getting valid 2 point contours 
            if self._check_lines(cont_1) and self._check_lines(cont_2):
                pass
            else: 
                # TODO: skip and continue instead of just breaking??
                sys.exit("Invalid contour(s)")
                
            # Now order the contour points correcly
            if cont_1[0][0] < cont_2 [0][0]:
                # Cont 1 is left finger
                left_fing_line = cont_1
                right_fing_line = cont_2
            else:
                # Cont 2 is the left finger
                right_fing_line = cont_1
                left_fing_line = cont_2    
                
            # Return the endpoints of each finger line    
            return left_fing_line, right_fing_line 
        except:
            return None, None
    

    def convert_pix_to_m(self, depth_image, left_finger, right_finger):
        pass

            
    def _check_lines(self, cont):
        """ Takes in a contour and checks that is a 2x2 array
        
        """
        if cont.shape == (2,2):
            return True
        else:
            return False

        
        output3 = color_image.copy()



           # Get pointcloud from depth image
        points = pc.calculate(aligned_depth_frame)

        # Get individual points from pointcloud

        vtx = np.asanyarray(points.get_vertices()).reshape(480, 640)