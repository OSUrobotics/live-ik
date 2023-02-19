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
import importlib  
hand = importlib.import_module("hand-gen-IK")


class ik_manager:

    def __init__(self):
        
        # General camera parameters
        self.camera_calibration = np.array(((587.65822288, 0.0, 312.22279429),(0.0, 587.25425585, 242.52669574),(0.0, 0.0, 1.00000000)))
        self.r_t_dists = np.array((.0744065755, .144374443, -.000463894288, -.00363146720, -1.13198957))

        self.initial_pose = [0.0, 0.0, 0.0] # Stores the first pose to use for relative calculations

        # Defining the asterisk directions for a standard 39 mm object
        #"SW": np.array([-0.135, -.0433]),
        #SW": np.array([-0.165, -.0433]),
        self.f1_direction_dict = {
            "N": np.array([0.015, .1567]),
            "NE": np.array([0.065, .1567]),
            "E": np.array([0.165, .1067]),
            "SE": np.array([.165, -.0433]),
            "S": np.array([0.015, -.0433]),
            "SW": np.array([-0.135, -.0433]),
            "W": np.array([-0.135, .1067]),
            "NW": np.array([-0.035, .1567])}
        self.f2_direction_dict = {
            "N": np.array([-0.015, .1567]),
            "NE": np.array([.035, .1567]),
            "E": np.array([0.135, .1067]),
            "SE": np.array([0.135, -.0433]),
            "S": np.array([-0.015, -.0433]),
            "SW": np.array([-0.165, -.0433]),
            "W": np.array([-0.165, .1067]),
            "NW": np.array([-0.065, .1567])}

        # Store the distace of the object from the palm (in y)
        self.palm_shift = .1067 # .1 m from object to palm
    
    def live_run(self, direction = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"], image_wait: int = 20):

        # Ok, so we start by setting up the classes
        ## ARUCO TRACKING
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        ## CONTOUR FINDING
        contour = ContourFind()
        ## CONTACT CALCULATIONS
        contact = ContactPoint()
        # INVERSE KINEMATICS
        testhand = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]], "offset": [.029, 0, 0]},
            "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]], "offset": [-.029, 0, 0]}}
        ik_left = hand.liveik.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger2"])
        ik_right = hand.liveik.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger1"])

        ## DYNAMIXEL setup
        self.dyn_replay_setup(hand_type="2v2")
        self.dynamixel_control.update_PID(85,25,45)
        self.dynamixel_control.update_speed(500)


        # Move Dynamixels to starting position
        self.dynamixel_control.go_to_initial_position()
        sleep(1)
        self.dynamixel_control.update_speed(50)

        # Wait for user input to start
        input("enter to continue")
        # Start RealSense
        at.start_realsense()

        first_time = True
        frame_counter = 0 
        while True:
            # Get the color image and point data
            color_image, vtx = at.get_frame()

            if color_image is None or vtx is None:
                # Check that we actually recived an image and points
                continue

            # Wait for the first predetermined number of frames before performing calculations
            if frame_counter < 20:
                frame_counter+=1
                continue

            # Get our current object pose in pixel coordinates
            current_pose, _, _ = at.object_pose(color_image, vtx, True)
            if not current_pose.any():
                # If unable to determine a pose, continue
                continue
        
            # Get the contours back in pixel coordinates
            f_l_contour, f_r_contour, orig_c_left, orig_c_right = contour.find_countours(color_image)

              
            # Convert from from pixel coordinates to m w/ depth data
            object_pose = self._pix_to_m(current_pose[0:2], vtx)
            if first_time:
                # If this is the first frame we are capturing, save this as our intial position to use for relative calculations
                first_time = False
                self.initial_pose = object_pose
                continue

            finger_l_contour_m = self._pix_to_m(f_l_contour, vtx)
            finger_r_contour_m = self._pix_to_m(f_r_contour, vtx)
                        # Get the current motor positions
            self.dynamixel_control.bulk_read_pos()  # Read the current motor positions
            m0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right bottom
            m1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 - right top
            m2 = self.dynamixel_control.dxls[2].read_position_m # Get the position of motor 2 - left bottom
            m3 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 3 - left top

            joint_right = [m0, m1]
            joint_left = [m2, m3]
            # Update our angles in the FK with current motor angles
            ik_left.update_angles = joint_left
            ik_right.update_angles = joint_right

            # Take the contours and object pose and calculate contact points
            contact_point_l, contact_delta_l = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_l_contour_m, [m2, m3], "L")
            contact_point_r, contact_delta_r = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_r_contour_m, [m0, m1], "R")

            contact_delta_l[1] = min(contact_delta_l[1], .072)
            contact_delta_r[1] = min(contact_delta_r[1], .072)

            # For plotting, calculate the pixels per mm
            test_obj = np.array([current_pose[0]+10, current_pose[1]])
            test_obj_mm = self._pix_to_m(test_obj, vtx)
            diff_x = test_obj_mm[0] - object_pose[0] # Distance per 10 pixels in x

            test_obj = np.array([current_pose[0], current_pose[1]-10])
            test_obj_mm = self._pix_to_m(test_obj, vtx)
            diff_y = test_obj_mm[1] - object_pose[1] # Distance per 10 pixels in y

            # Check that we have valid pixels per mm
            if np.isclose(diff_x, 0.0) or np.isclose(diff_y, 0.0):
                continue

            # Now take a find the right contact point's number of pixels 
            x_r = int(10*(object_pose[0]-contact_point_r[0])/diff_x) 
            y_r = int(10*(object_pose[1]-contact_point_r[1])/diff_y)

            # Now take and find the left contact point's number of pixels 
            x_l = int(10*(object_pose[0]-contact_point_l[0])/diff_x) 
            y_l = int(10*(object_pose[1]-contact_point_l[1])/diff_y)


            show_image = False
            if show_image:
                # Draw contours        
                contour_image = cv2.drawContours(color_image, [orig_c_left, orig_c_right], -1, (0, 255, 0), 3)
                
                # Draw a red circle with zero radius and -1 for filled circle
                image2 = cv2.circle(color_image, (int(current_pose[0]-x_l),int(current_pose[1]+y_l)), radius=3, color=(0, 0, 255), thickness=-1)
                image3 = cv2.circle(color_image, (int(current_pose[0]-x_r),int(current_pose[1]+y_r)), radius=3, color=(255, 0, 0), thickness=-1)

                cv2.imshow("hi", image3)
                cv2.waitKey(5)
                
            
            # Calculate the relative pose for the IK solver
            # Object starts 10cm from the joints

            # We just need to pass in the joint angles and the contact point deltas

            # Left finger has joints 0 and 1



            # We need to get the target
            """
            Target is a point on the line from contact point to goal
            Where 0,0 is palm center

            So we need to:
            1) Get point along line from contact to goal
            2) Take that point and translate it the amount of the starting position of the object (to get it relative to 0,0 of starting object)
            3) Translate the point 10 cm in y (get relative to palm base, not initial object position)
            """
            # Calculate the target point for the left finger
            shifted_by_start_l = [contact_point_l[0]-self.initial_pose[0], contact_point_l[1]-self.initial_pose[1]]
            shifted_by_palm_l = [shifted_by_start_l[0], shifted_by_start_l[1]+self.palm_shift]
            l_point = self.step_towards_goal(shifted_by_palm_l, self.f2_direction_dict["SE"], .015) # was .02 with smoothing
            

            #l_point = self.step_towards_goal(contact_point_l, self.f1_direction_dict["E"], .02)
            #shifted_by_start_l = [l_point[0]-self.initial_pose[0], l_point[1]-self.initial_pose[1]]
            #shifted_by_palm_l = [shifted_by_start_l[0], shifted_by_start_l[1]+self.palm_shift]

            # # Calculate the target point for the right finger
            shifted_by_start_r = [contact_point_r[0]-self.initial_pose[0], contact_point_r[1]-self.initial_pose[1]]
            shifted_by_palm_r = [shifted_by_start_r[0], shifted_by_start_r[1]+self.palm_shift]
            r_point = self.step_towards_goal(shifted_by_palm_r, self.f1_direction_dict["SE"], .015)

            
            # Calculate the inverse kinematics for each finger
            print(f"L: {contact_delta_l}, R: {contact_delta_r}")
            _, new_angles_l, num_itl = ik_left.calculate_ik(target = l_point, ee_location=[contact_delta_l[0], contact_delta_l[1], 1])
            _, new_angles_r, num_itr = ik_right.calculate_ik(target = r_point, ee_location=[contact_delta_r[0], contact_delta_r[1], 1])
            print(F"Num L: {num_itl}, Num R: {num_itr}")
            print(f"FK: {ik_right.finger_fk.calculate_forward_kinematics()}")
            
            print(f"m0: {m0}, new_m0: {new_angles_r[0]}::: m1: {m1}, new_m1: {new_angles_r[1]}::: m2: {m2}, new_m2: {new_angles_l[0]}::: m3: {m3}, new_m3: {new_angles_l[1]}")
            print(f"Contact left: {shifted_by_palm_l}, Contact right: {shifted_by_palm_r}")
            print(f"Target left: {l_point}, Target Right: {r_point}")
            #print(joint_a)
            limit = .6
            if np.abs(new_angles_r[0]-m0) > limit or np.abs(new_angles_r[1]-m1) > limit or np.abs(new_angles_l[0]-m2) > limit or np.abs(new_angles_l[1]-m3) > limit:
            #     if new_angles_r[0] > 0:
            #         new_angles_r[0] = min(new_angles_r[0], m0+limit)
            #     else:
            #         new_angles_r[0] = max(new_angles_r[0], m0-limit)

            #     if new_angles_r[1] > 0:
            #         new_angles_r[1] = min(new_angles_r[1], m1+limit)
            #     else:
            #         new_angles_r[1] = max(new_angles_r[1], m1-limit)

            #     if new_angles_l[0] > 0:
            #         new_angles_l[0] = min(new_angles_l[0], m2+limit)
            #     else:
            #         new_angles_l[0] = max(new_angles_l[0], m2-limit)

            #     if new_angles_l[1] > 0:
            #         new_angles_l[1] = min(new_angles_l[1], m3+limit)
            #     else:
            #         new_angles_l[1] = max(new_angles_l[1], m3-limit)

                print("Bad value")
                continue
                
            #print(f"Left Contact: {contact_point_l}, Left New: {l_point}")
            #print(f"m0: {m0}, m1: {m1}, new_m0: {new_angles_l[0]}, new_m1: {new_angles_l[1]}")
            #
            # TODO: Add check that verifies the differnce between all of the angles and the old is not excessive
            #print(f"m0: {m0}, m1: {m1}, new_m0: {new_angles_l[0]}, new_m1: {new_angles_l[1]}")
            num = 1
            goal0 = new_angles_r[0]#np.linspace(m0, new_angles_r[0], num)
            goal1 = new_angles_r[1] #np.linspace(m1, new_angles_r[1], num)
            goal2 = new_angles_l[0] #np.linspace(m2, new_angles_l[0], num)
            goal3 = new_angles_l[1] #np.linspace(m3, new_angles_l[1], num)
            for i in range(num):
                # Update all the positions with the following: center_position + difference in 0-1023 scale
                self.dynamixel_control.update_goal(0, self.dynamixel_control.dxls[0].center_pos+self.dynamixel_control.convert_rad_to_pos(goal0))#+self.dynamixel_control.dxls[0].shift)
                self.dynamixel_control.update_goal(1, self.dynamixel_control.dxls[1].center_pos+self.dynamixel_control.convert_rad_to_pos(goal1))#+self.dynamixel_control.dxls[1].shift)
                self.dynamixel_control.update_goal(2, self.dynamixel_control.dxls[2].center_pos+self.dynamixel_control.convert_rad_to_pos(goal2))#+self.dynamixel_control.dxls[2].shift)
                self.dynamixel_control.update_goal(3, self.dynamixel_control.dxls[3].center_pos+self.dynamixel_control.convert_rad_to_pos(goal3))#+self.dynamixel_control.dxls[3].shift)

                print(f"Goal: {self.dynamixel_control.dxls[0].goal_position}, Previous Reading: {self.dynamixel_control.dxls[0].read_position}")

                #tes = self.dynamixel_control.dxls[0].goal_position
                #print(f"Old goal: {tes}, New goal: {self.dynamixel_control.dxls[0].goal_position}, Current pos: {m0}")
                self.dynamixel_control.send_goal()
            
            # Read dynamixel position and wait until within 5% of goal
            while True:
                self.dynamixel_control.bulk_read_pos()  # Read the current motor positions
                current_0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right bottom
                current_1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 - right top
                current_2 = self.dynamixel_control.dxls[2].read_position_m # Get the position of motor 2 - left bottom
                current_3 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 3 - left top
                # m0 is old position
                # goal 0 is new target
                # current 0 is current position reading 
                diff_0 = np.abs(goal0-current_0)/(np.abs(goal0-m0))
                diff_1 = np.abs(goal1-current_1)/(np.abs(goal1-m1))
                diff_2 = np.abs(goal2-current_2)/(np.abs(goal2-m2))
                diff_3 = np.abs(goal3-current_3)/(np.abs(goal3-m3))

                if max(diff_0, diff_1, diff_2, diff_3) < 5.0:
                    break



            

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



    def dyn_replay_setup(self, hand_type = "2v2"):
        self.dynamixel_control = dynamixel_control.Dynamixel()

        if hand_type == "2v2":
            self.dynamixel_control.add_dynamixel(ID_number=0, calibration=[0, 465, 1023], shift = 25) # Negative on left side was -25
            self.dynamixel_control.add_dynamixel(ID_number=1, calibration=[0, 545, 1023], shift = 0)
            self.dynamixel_control.add_dynamixel(ID_number=2, calibration=[0, 450, 1023], shift = -25) # Positive on right side was 25
            self.dynamixel_control.add_dynamixel(ID_number=3, calibration=[0, 553, 1023], shift = 0)
        elif hand_type == "3v3":
            print("not implemented")
            pass

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