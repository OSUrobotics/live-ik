import aruco_tracking as aruco
import contact_calculation 
from contour_find import ContourFind
import cv2
import numpy as np
import threading
import logging
import dynamixel_control
from time import time, sleep
import sys
from contact_calculation import ContactPoint
import os
from handGenIK import liveik as IK

class ik_manager:

    def __init__(self):
        
        # General camera parameters
        self.camera_calibration = np.array(((587.65822288, 0.0, 312.22279429),(0.0, 587.25425585, 242.52669574),(0.0, 0.0, 1.00000000)))
        self.r_t_dists = np.array((.0744065755, .144374443, -.000463894288, -.00363146720, -1.13198957))
        #old self.camera_calibration = np.array(((591.40261976, 0.0, 323.94871535),(0.0, 593.59306833, 220.0225822),(0.0, 0.0, 1.00000000)))
        #old self.r_t_dists = np.array((0.07656341,  0.41328222, -0.02156859,  0.00270287, -1.64179927))

        self.initial_pose = [0.0, 0.0, 0.0]

        self.f1_direction_dict = {
            "N": np.array([0.01, .15]),
            "NE": np.array([0.2, .2]),
            "E": np.array([0.2, .1067]),
            "SE": np.array([.2, 0]),
            "S": np.array([0.01, .05]),
            "SW": np.array([-0.18, 0]),
            "W": np.array([-0.18, .1067]),
            "NW": np.array([-0.18, .2])}
        self.f2_direction_dict = {
            "N": np.array([-0.01, .15]),
            "NE": np.array([0.18, .2]),
            "E": np.array([0.18, .1067]),
            "SE": np.array([0.18, 0]),
            "S": np.array([-0.01, .05]),
            "SW": np.array([-0.2, 0]),
            "W": np.array([-0.2, .1067]),
            "NW": np.array([-0.2, .2])}

        self.palm_shift = .1 # .1 m from object to palm
    
    def live_run(self):
        # Set up the IK
        testhand = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]], "offset": [.03, 0, 0]},
            "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]], "offset": [-.03, 0, 0]}}
        ik_left = IK.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger2"])
        ik_right = IK.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger1"])

        # Now 


        # Ok, so we start by setting up the classes
        ## ARUCO
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        ## CONTOUR
        contour = ContourFind()
        ## CONTACT
        contact = ContactPoint()
        ## DYNAMIXEL
        self.dyn_replay_setup()

        # Move Dynamixels to starting position
        self.dynamixel_control.go_to_initial_position()

        #input("Press enter to coninue.")
        #print("You have 3 seconds to reset")
        input("enter to continue")
        # Start RealSense
        at.start_realsense()
        #sleep(1)
        first_time = True
        first_counter = 0 
        while True:
            # Get the color image and point data
            color_image, vtx = at.get_frame()

            if first_counter < 20:
                first_counter+=1
                continue

            # Get our current object pose in pixel coordinates
            current_pose, corners, ids = at.object_pose(color_image, vtx, True)
            if not current_pose.any():
                continue
            
        
            # Get the contours back in pixel coordinates
            f1_contour, f2_contour, orig_c1, orig_c2 = contour.find_countours(color_image)
            if f1_contour is not None:
                contour_image = cv2.drawContours(color_image, [orig_c1, orig_c2], -1, (0, 255, 0), 3)
                

            # Convert from from pixel coordinates to m w/ depth data
            object_pose = self._pix_to_m(current_pose[0:2], vtx)
            if first_time:
                first_time = False
                self.initial_pose = object_pose
                continue

            finger_1_contour_m = self._pix_to_m(f1_contour, vtx)
            finger_2_contour_m = self._pix_to_m(f2_contour, vtx)

            # Take the contours and object pose and calculate contact points
            contact_point_l, contact_delta_l = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_1_contour_m)
            contact_point_r, contact_delta_r = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_2_contour_m)

            # For plotting, calculate the pixels per mm
            test_obj = np.array([current_pose[0]+10, current_pose[1]])
            test_obj_mm = self._pix_to_m(test_obj, vtx)
            diff_x = test_obj_mm[0] - object_pose[0] # Distance per 10 pixels in x

            test_obj = np.array([current_pose[0], current_pose[1]-10])
            test_obj_mm = self._pix_to_m(test_obj, vtx)
            diff_y = test_obj_mm[1] - object_pose[1] # Distance per 10 pixels in y

            # Now take a find the contact point's number of pixels 
            if np.isclose(diff_x, 0.0) or np.isclose(diff_y, 0.0):
                continue
            x_l = int(10*(object_pose[0]-contact_point_l[0])/diff_x) 
            y_l = int(10*(object_pose[1]-contact_point_l[1])/diff_y)
            #print(f"X_l: {x_l}, object: {current_pose[0]}, combined: {x_l +current_pose[0]}")
            #print(f"Y: {y_l}, object: {current_pose[1]}, combined: {y_l +current_pose[1]}")
            # Now take a find the contact point's number of pixels 
            x_r = int(10*(object_pose[0]-contact_point_r[0])/diff_x) 
            y_r = int(10*(object_pose[1]-contact_point_r[1])/diff_y)
            #print(f"X_l: {x_r}, object: {current_pose[0]}, combined: {x_r +current_pose[0]}")
            #print(f"Y: {y_r}, object: {current_pose[1]}, combined: {y_r +current_pose[1]}")

            #Draw a red circle with zero radius and -1 for filled circle
            image2 = cv2.circle(color_image, (int(current_pose[0]-x_l),int(current_pose[1]+y_l)), radius=3, color=(0, 0, 255), thickness=-1)
            image3 = cv2.circle(color_image, (int(current_pose[0]-x_r),int(current_pose[1]+y_r)), radius=3, color=(255, 0, 0), thickness=-1)

            cv2.imshow("hi", image3)
            cv2.waitKey(500)
            # Calculate the relative pose for the IK solver
            # Object starts 10cm from the joints

            # We just need to pass in the joint angles and the contact point deltas

            # Left finger has joints 0 and 1

            
            self.dynamixel_control.bulk_read_pos()  # Read the current motor positions
            m0 = self.dynamixel_control.dxls[0].read_position # Get the position of motor 0
            m1 = self.dynamixel_control.dxls[1].read_position # Get the position of motor 1
            m2 = self.dynamixel_control.dxls[2].read_position # Get the position of motor 2
            m3 = self.dynamixel_control.dxls[3].read_position # Get the position of motor 3

            joint_a = [m2, m3, m0, m1]
            # Update our angles in the FK 
            ik_left.update_angles = joint_a
            ik_right.update_angles = joint_a


            # We need to get the target
            """
            Target is a point on the line from contact point to goal
            Where 0,0 is palm center

            So we need to:
            1) Get point along line from contact to goal
            2) Take that point and translate it the amount of the starting position of the object (to get it relative to 0,0 of starting object)
            3) Translate the point 10 cm in y (get relative to palm base, not initial object position)
            """
            l_point = self.step_towards_goal(contact_point_l, self.f1_direction_dict["E"], .025)
            shifted_by_start_l = [l_point[0]-self.initial_pose[0], l_point[1]-self.initial_pose[1]]
            shifted_by_palm_l = [shifted_by_start_l[0], shifted_by_start_l[1]+self.palm_shift]

            r_point = self.step_towards_goal(contact_point_r, self.f2_direction_dict["E"], .025)
            shifted_by_start_r = [r_point[0]-self.initial_pose[0], r_point[1]-self.initial_pose[1]]
            shifted_by_palm_r = [shifted_by_start_r[0], shifted_by_start_r[1]+self.palm_shift]
            #p#rint(f"Contact: {contact_point_l}, Shifted {shifted_by_palm_l}")
            #p#rint(f"Contact in frame: {contact_delta_l}")

            
            #sleep(5)
            # Now we calculate the ik
            
            _, new_angles_l, _ = ik_left.calculate_ik(target = shifted_by_palm_l, ee_location=[contact_delta_l[0], contact_delta_l[1], 1])
            _, new_angles_r, _ = ik_right.calculate_ik(target = shifted_by_palm_r, ee_location=[contact_delta_r[0], contact_delta_r[1], 1])
            #print(new_angles_l)
            print(f"m0: {m0}, m1: {m1}, new_m0: {new_angles_l[0]}, new_m1: {new_angles_l[1]}")
            #print(f"m2: {m2}, m3: {m3}, new_m2: {new_angles_r[0]}, new_m3: {new_angles_r[1]}")
            
            #ik_right.calculate_ik(target = 0, ee_location=[contact_delta_r[0], contact_delta_r[1], 1])
            #print(f"M0: {m0}, M1: {m1}")

            #print(f"Contact l: {contact_delta_l}, Contact r: {contact_delta_r}")

            tes = self.dynamixel_control.dxls[0].goal_position
            
            
            self.dynamixel_control.update_goal(0, self.dynamixel_control.dxls[0].center_pos+self.dynamixel_control.convert_rad_to_pos(new_angles_l[0]))
            self.dynamixel_control.update_goal(1, self.dynamixel_control.dxls[1].center_pos+self.dynamixel_control.convert_rad_to_pos(new_angles_l[1]))
            self.dynamixel_control.update_goal(2, self.dynamixel_control.dxls[2].center_pos+self.dynamixel_control.convert_rad_to_pos(new_angles_r[0]))
            self.dynamixel_control.update_goal(3, self.dynamixel_control.dxls[3].center_pos+self.dynamixel_control.convert_rad_to_pos(new_angles_r[1]))
            print(f"Old goal: {tes}, New goal: {self.dynamixel_control.dxls[0].goal_position}, Current pos: {m0}")
            self.dynamixel_control.send_goal()
            


            
            
            #print(finger_1_contour_m)
            
                # Save our intiial pose of the object
                
            #if not dy.is_alive():
            #    print("Finished moving")
            #    break

    def step_towards_goal(self, start_vec, end_vec, distance):
        temp_x = end_vec[0] - start_vec[0]
        temp_y = end_vec[1] - start_vec[1]
        magnitude = np.sqrt((temp_x**2 + temp_y**2))
        if magnitude <= distance:
            return [end_vec[0], end_vec[1]]
        temp_x /= magnitude
        temp_y /= magnitude
        temp_x = start_vec[0] + distance*temp_x
        temp_y = start_vec[1] + distance*temp_y
        return [temp_x, temp_y]



    def dyn_replay_setup(self):
        self.dynamixel_control = dynamixel_control.Dynamixel()
        self.dynamixel_control.add_dynamixel(self.dynamixel_control.create_dynamixel_dict(ID_number=0, calibration=[0, 450, 1023], shift = -25)) # Negative on left side
        self.dynamixel_control.add_dynamixel(self.dynamixel_control.create_dynamixel_dict(ID_number=1, calibration=[0, 553, 1023], shift = 0))
        self.dynamixel_control.add_dynamixel(self.dynamixel_control.create_dynamixel_dict(ID_number=2, calibration=[0, 465, 1023], shift = 25)) # Positive on right side
        self.dynamixel_control.add_dynamixel(self.dynamixel_control.create_dynamixel_dict(ID_number=3, calibration=[0, 545, 1023], shift = 0))

        self.dynamixel_control.setup_all()
        
    def dyn_replay(self, dyn_file_location="Open_Loop_Data", dyn_file_name="angles_E.pkl", delay = .005):
        self.dynamixel_control.replay_pickle_data(dyn_file_location, dyn_file_name, delay_between_steps = delay)

    def dyn_replay_thread(self, dyn_file_location="Open_Loop_Data", dyn_file_name="angles_E.pkl", delay = .005):
        dy = threading.Thread(target=self.dyn_replay, args=(dyn_file_location,dyn_file_name,delay,), daemon=True)
        dy.start()

    def linear_run(self, pickle_file):
        # Ok, so we start by setting up the classes
        ## ARUCO
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        ## CONTOUR
        contour = ContourFind()
        ## CONTACT
        contact = ContactPoint()
        ## DYNAMIXEL
        self.dyn_replay_setup()

        # Move Dynamixels to starting position
        self.dynamixel_control.go_to_initial_position()

        #input("Press enter to coninue.")
        print("You have 3 seconds to reset")
        sleep(3)
        # Start RealSense
        at.start_realsense()
        #sleep(1)
        first_time = True
        while True:
            # Get the color image and point data
            color_image, vtx = at.get_frame()

            # Get our current object pose in pixel coordinates
            current_pose, corners, ids = at.object_pose(color_image, vtx, True)
            if not current_pose.any():
                continue
            
        
            # Get the contours back in pixel coordinates
            f1_contour, f2_contour, orig_c1, orig_c2 = contour.find_countours(color_image)
            if f1_contour is not None:
                contour_image = cv2.drawContours(color_image, [orig_c1, orig_c2], -1, (0, 255, 0), 3)
                

            # Convert from from pixel coordinates to m w/ depth data
            object_pose = self._pix_to_m(current_pose[0:2], vtx)
            finger_1_contour_m = self._pix_to_m(f1_contour, vtx)
            finger_2_contour_m = self._pix_to_m(f2_contour, vtx)

            # Take the contours and object pose and calculate contact points
            contact_point_l, contact_delta_l = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_1_contour_m)
            contact_point_r, contact_delta_r = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_2_contour_m)

            # For plotting, calculate the pixels per mm
            test_obj = np.array([current_pose[0]+10, current_pose[1]])
            test_obj_mm = self._pix_to_m(test_obj, vtx)
            diff_x = test_obj_mm[0] - object_pose[0] # Distance per 10 pixels in x

            test_obj = np.array([current_pose[0], current_pose[1]-10])
            test_obj_mm = self._pix_to_m(test_obj, vtx)
            diff_y = test_obj_mm[1] - object_pose[1] # Distance per 10 pixels in y

            # Now take a find the contact point's number of pixels 
            if np.isclose(diff_x, 0.0) or np.isclose(diff_y, 0.0):
                continue
            x_l = int(10*(object_pose[0]-contact_point_l[0])/diff_x) 
            y_l = int(10*(object_pose[1]-contact_point_l[1])/diff_y)
            #print(f"X_l: {x_l}, object: {current_pose[0]}, combined: {x_l +current_pose[0]}")
            #print(f"Y: {y_l}, object: {current_pose[1]}, combined: {y_l +current_pose[1]}")
            # Now take a find the contact point's number of pixels 
            x_r = int(10*(object_pose[0]-contact_point_r[0])/diff_x) 
            y_r = int(10*(object_pose[1]-contact_point_r[1])/diff_y)
            #print(f"X_l: {x_r}, object: {current_pose[0]}, combined: {x_r +current_pose[0]}")
            #print(f"Y: {y_r}, object: {current_pose[1]}, combined: {y_r +current_pose[1]}")

            #Draw a red circle with zero radius and -1 for filled circle
            image2 = cv2.circle(color_image, (int(current_pose[0]-x_l),int(current_pose[1]+y_l)), radius=3, color=(0, 0, 255), thickness=-1)
            image3 = cv2.circle(color_image, (int(current_pose[0]-x_r),int(current_pose[1]+y_r)), radius=3, color=(255, 0, 0), thickness=-1)

            cv2.imshow("hi", image3)
            cv2.waitKey(1)
            #print(finger_1_contour_m)
            if first_time:
                first_time = False
                dy = threading.Thread(target=self.dyn_replay, args=("Open_Loop_Data", pickle_file, .005,), daemon=True)
                dy.start()
            if not dy.is_alive():
                print("Finished moving")
                break


    def _pix_to_m(self, input, vtx):
        # Conver all inputed values to mm. Will work from a single val up to 2 levels 
        
        in_shape = input.shape
        if in_shape == (2,):
            converted = np.zeros(in_shape, dtype=float)
            converted = [vtx[int(input[1])][int(input[0])][0], -vtx[int(input[1])][int(input[0])][1]]
        else:
            converted = np.zeros(input.shape, dtype=float)
            for i, val in enumerate(input):
                # vtx[x][y][x or y or z (1-3)]
                # x,y in mm
                # vtx is y,x
                converted[i] = [vtx[int(val[1])][int(val[0])][0], -vtx[int(val[1])][int(val[0])][1]]

        return converted


    def contour_visualizer(self):
        # Ok, so we start by setting up the classes
        ## ARUCO
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        ## CONTOUR
        contour = ContourFind()
        ## DYNAMIXEL
        #self.dyn_replay_setup()

        # Move Dynamixels to starting position
        #self.dynamixel_control.go_to_initial_position()

        #input("Press enter to coninue.")
        
        # Start RealSense
        at.start_realsense()

        while True:
            # Get the color image and point data
            color_image, vtx = at.get_frame()

            # Get our current object pose in pixel coordinates
            current_pose, corners, ids = at.object_pose(color_image, vtx, True)
            if not current_pose.any():
                continue
            
        
            # Get the contours back in pixel coordinates
            f1_contour, f2_contour, orig_c1, orig_c2 = contour.find_countours(color_image)
            if f1_contour is not None:
                contour_image = cv2.drawContours(color_image, [orig_c1, orig_c2], -1, (0, 255, 0), 3)


            cv2.imshow("hi", contour_image)
            cv2.waitKey(1)
            #print(finger_1_contour_m)
            '''
            if first_time:
                first_time = False
                dy = threading.Thread(target=self.dyn_replay, args=("Open_Loop_Data", "angles_W.pkl",.005,), daemon=True)
                dy.start()
            if not dy.is_alive():
                print("Finished moving")
                break
            '''



if __name__ == "__main__":
    manager = ik_manager()

    """
    pickle_files = ["angles_N.pkl", "angles_NE.pkl", "angles_E.pkl", "angles_SE.pkl", "angles_S.pkl", "angles_SW.pkl", "angles_W.pkl", "angles_NW.pkl"]
    for pkl in pickle_files:
        manager.linear_run(pkl)
    #manager.contour_visualizer()
    """
    
    
    manager.live_run()