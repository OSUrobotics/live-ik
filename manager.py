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
import pickle as pkl
import os
from copy import deepcopy
import importlib  
hand = importlib.import_module("hand-gen-IK")

class ik_manager:

    def __init__(self):
        
        # General camera parameters
        self.camera_calibration = np.array(((587.65822288, 0.0, 312.22279429),(0.0, 587.25425585, 242.52669574),(0.0, 0.0, 1.00000000)))
        self.r_t_dists = np.array((.0744065755, .144374443, -.000463894288, -.00363146720, -1.13198957))

        self.aruco_params = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                            "aruco_params": cv2.aruco.DetectorParameters_create(),
                            "marker_side_dims": 0.04,
                            "opencv_camera_calibration": self.camera_calibration,
                            "opencv_radial_and_tangential_dists": self.r_t_dists
                            }

        self.initial_pose = [0.0, 0.0, 0.0] # Stores the first pose to use for relative calculations

        # Defining the asterisk directions for a standard 39 mm object
        self.f1_direction_dict = {
            "N": np.array([0.0, .10]), # done
            "NE": np.array([0.15, .15]),
            "E": np.array([0.15, 0.0]),
            "SE": np.array([0.15, -.15]),
            "S": np.array([0.0, -.15]),
            "SW": np.array([-0.15, -.15]),
            "W": np.array([-0.15, 0.0]),
            "NW": np.array([-.15, .15])}
        self.f2_direction_dict = {
            "N": np.array([0.0, .10]), # done
            "NE": np.array([0.15, .15]),
            "E": np.array([0.15, 0.00]),
            "SE": np.array([0.15, -.15]),
            "S": np.array([0.0, -.15]),
            "SW": np.array([-0.15, -.15]),
            "W": np.array([-0.15, 0.0]),
            "NW": np.array([-.15, .15])}
        
        """
        self.f1_direction_dict = {
            "N": np.array([0.0225, .31005]), # done
            "NE": np.array([0.1725, .31005]),
            "E": np.array([0.1725, .16005]),
            "SE": np.array([.1725, .01005]),
            "S": np.array([0.0225, .01005]),
            "SW": np.array([-.1275, .01005]),
            "W": np.array([-.1275, .16005]),
            "NW": np.array([-.1275, .26005])}
        self.f2_direction_dict = {
            "N": np.array([-0.0225, .31005]),
            "NE": np.array([0.1275, .31005]),
            "E": np.array([0.1275, .16005]),
            "SE": np.array([0.1275, .01005]),
            "S": np.array([-0.0225, .01005]),
            "SW": np.array([-0.1725, .01005]),
            "W": np.array([-0.1725, .16005]),
            "NW": np.array([-0.1725, .31005])}
        self.f1_direction_dict = {
            "N": np.array([0.015, .1567]),
            "NE": np.array([0.065, .1567]),
            "E": np.array([0.165, .1067]),
            "SE": np.array([.165, -.0433]),
            "S": np.array([0.015, .02]),
            "SW": np.array([-0.135, -.0433]),
            "W": np.array([-0.135, .1067]),
            "NW": np.array([-0.035, .1567])}
        self.f2_direction_dict = {
            "N": np.array([-0.015, .1567]),
            "NE": np.array([.035, .1567]),
            "E": np.array([0.135, .1067]),
            "SE": np.array([0.135, -.0433]),
            "S": np.array([-0.015, .02]),
            "SW": np.array([-0.165, -.0433]),
            "W": np.array([-0.165, .1067]),
            "NW": np.array([-0.065, .1567])}

        """

        # Store the distace of the object from the palm (in y)
        self.palm_shift = .16 # .1 m from object to palm

        # Set parameters
        self.left_dist_length = .072
        self.right_dist_length = .072
        self.left_sleeve_length = .050
        self.right_sleeve_length = .050

        self.move_complete = True
        self.done = False
        self.block = False # Blocking variable to prevent reading at the same time as writing
        self.event = threading.Event()
    
    
    def live_run(self, direction = "NW", hand_name = "2v2", ratios="1.1_1.1_1.1_1.1", trial="1"):
        # Ok, so we start by setting up the classes
        ## ARUCO TRACKING
        at = aruco.Aruco_Track(self.aruco_params)
        ## CONTOUR FINDING
        contour = ContourFind()
        ## CONTACT CALCULATIONS
        contact = ContactPoint(object_size=58.5)

        
        fil_name= hand_name + "_" + ratios
        load_name = direction+"_"+fil_name+".pkl"
        file_name = direction+"_"+fil_name+"_"+trial+".pkl"
        folder_name = os.path.join("Open_Loop_Data", fil_name)
        #folder_name_init = os.path.join("Open_Loop_Data", name)

        # INVERSE KINEMATICS
        if hand_name == "2v2":
            
            testhand = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .108, 0], [0, .108, 0]], "offset": [.047, 0, 0]},
                "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .108, 0], [0, .108, 0]], "offset": [-.047, 0, 0]}}
        elif hand_name == "2v3":
            testhand = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .054, 0], [0, .162, 0]], "offset": [.04, 0, 0]},
                "finger2": {"name": "finger1", "num_links": 3, "link_lengths": [[0, .054, 0], [0, .0756, 0], [0, .0864, 0]], "offset": [-.04, 0, 0]}}
        elif hand_name == "3v3":
            testhand = {"finger1": {"name": "finger0", "num_links": 3, "link_lengths": [[0, .0972, 0], [0, .054, 0], [0, .06479, 0]], "offset": [.05, 0, 0]},
                "finger2": {"name": "finger1", "num_links": 3, "link_lengths": [[0, .0756, 0], [0, .054, 0], [0, .0864, 0]], "offset": [-.05, 0, 0]}}
        ik_left = hand.liveik.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger2"])
        ik_right = hand.liveik.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger1"])

        # Update values
        self.palm_shift = .1675
        ## DYNAMIXEL setup
        self.dyn_setup(hand_type=hand_name)
        self.dynamixel_control.update_PID(85,40,45) # I term was 25
        self.dynamixel_control.update_speed(200)
        # Move Dynamixels to starting position
        self.dynamixel_control.go_to_initial_position(folder_name, load_name)
        sleep(1)
        self.dynamixel_control.update_speed(50)
        # Wait for user input to start
        input("enter to continue")

        # Start RealSense
        at.start_realsense()    
        
        #os.path.dirname(__file__))
        #file_path = os.path.join(path_to, file_location, file_name)
    
        #with open(file_path, 'rb') as f:
        #    self.data = pkl.load(f)
        save_list = []

        first_time = True
        frame_counter = 0 
        try:
            while True:
                if self.done:
                    break

                # Get the color image and point data
                color_image, vtx = at.get_frame()

                if color_image is None or vtx is None:
                    # Check that we actually recived an image and points
                    continue

                # Wait for the first predetermined number of frames before performing calculations
                if frame_counter < 40:
                    frame_counter+=1
                    continue

                # Get our current object pose in pixel coordinates
                current_pose, _, _ = at.object_pose(color_image, vtx, True)
                if current_pose is None:
                    # If unable to determine a pose, continue
                    continue
            
                # Get the contours back in pixel coordinates
                f_l_contour, f_r_contour, orig_c_left, orig_c_right = contour.find_countours(color_image)
                if f_l_contour is None:
                    continue
                
                # Convert from from pixel coordinates to m w/ depth data
                object_pose = self._pix_to_m(current_pose[0:2], vtx)
                if first_time:
                    # If this is the first frame we are capturing, save this as our intial position to use for relative calculations
                    self.dynamixel_control.bulk_read_pos()   # Read the current motor positions
                    first_time = False
                    self.initial_pose = [object_pose[0], object_pose[1], current_pose[2]]
                    continue
                
                finger_l_contour_m = self._pix_to_m(f_l_contour, vtx)
                finger_r_contour_m = self._pix_to_m(f_r_contour, vtx)
                zero_array = np.zeros(2)
                if np.all(np.isclose(finger_l_contour_m[0], zero_array)) or np.all(np.isclose(finger_l_contour_m[1], zero_array)) or np.all(np.isclose(finger_r_contour_m[0], zero_array)) or np.all(np.isclose(finger_r_contour_m[1], zero_array)):
                    # We got a bad contour postion
                    print('Bad contour position!')
                    continue

                # Get the current motor positions 
                #self.dynamixel_control.bulk_read_pos()            
                m0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right prox
                m1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 
                m2 = self.dynamixel_control.dxls[2].read_position_m # Get the position of motor 2 
                m3 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 3
                m4 = None
                m5 = None
                if hand_name == "2v2":
                    joint_right = [m0, m1]
                    joint_left = [m2, m3]
                elif hand_name == "2v3":
                    m4 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 4 - left distal
                    joint_right = [m0, m1]
                    joint_left = [m2, m3, m4]
                elif hand_name == "3v3":
                    print("here")
                    m4 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 4 - left intermediate
                    m5 = self.dynamixel_control.dxls[5].read_position_m # Get the position of motor 5 - left distal
                    joint_right = [m0, m1, m2]
                    joint_left = [m3, m4, m5]
                #print(joint_left)
                #print(joint_right)

                


                # Take the contours and object pose and calculate contact points
                contact_point_l, contact_delta_l = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_l_contour_m, joint_left, "L", dist_length=self.left_dist_length, sleeve_length=self.left_sleeve_length)
                contact_point_r, contact_delta_r = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_r_contour_m, joint_right, "R", dist_length=self.right_dist_length, sleeve_length=self.right_sleeve_length)

                contact_delta_l[1] = min(contact_delta_l[1], self.left_dist_length)
                contact_delta_r[1] = min(contact_delta_r[1], self.right_dist_length)
                #contact_delta_l[1] = min(contact_delta_l[1], .072)
                #contact_delta_r[1] = min(contact_delta_r[1], .072)

                actual_pose = np.subtract([object_pose[0], object_pose[1], current_pose[2]], self.initial_pose)
                data_dict = {"obj_pos": [actual_pose[0], actual_pose[1], .05], "obj_or": actual_pose[2], "angles": {"joint_1": m0, "joint_2": m1, "joint_3": m2, "joint_4": m3, "joint_5": m4, "joint_6": m5}}
                save_list.append(data_dict)
                #print("looping")

                show_image = True
                if show_image:
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
                    # Draw contours        
                    contour_image = cv2.drawContours(color_image, [orig_c_left, orig_c_right], -1, (0, 255, 0), 3)
                    
                    # Draw a red circle with zero radius and -1 for filled circle
                    image2 = cv2.circle(color_image, (int(current_pose[0]-x_l),int(current_pose[1]+y_l)), radius=3, color=(0, 0, 255), thickness=-1)
                    image3 = cv2.circle(color_image, (int(current_pose[0]-x_r),int(current_pose[1]+y_r)), radius=3, color=(255, 0, 0), thickness=-1)
                    image4 = cv2.circle(color_image, (int(current_pose[0]),int(current_pose[1])), radius=3, color=(0, 255, 0), thickness=-1)




                    cv2.imshow("hi", image4)
                    cv2.waitKey(5)

                if self.move_complete and not self.done:
                    # Update our angles in the FK with current motor angles
                    self.dynamixel_control.bulk_read_pos()
                    m0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right prox
                    m1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 
                    m2 = self.dynamixel_control.dxls[2].read_position_m # Get the position of motor 2 
                    m3 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 3
                    #m4 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 4 - left intermediate
                    #m5 = self.dynamixel_control.dxls[5].read_position_m # Get the position of motor 5 - left distal
                    joint_right = [m0, m1]
                    joint_left = [m2, m3]

                    ik_left.update_angles(joint_left)
                    ik_right.update_angles(joint_right)
                    #print("Joint_r_1", ik_right.finger_fk.current_angles)
                    print("starting thread")
                    self.move_complete = False
                    #self.block = True
                    self.move_thread(hand_name, actual_pose, contact_point_l, direction, contact_point_r, ik_left, ik_right, contact_delta_l, contact_delta_r, m0, m1, m2, m3, m4, m5)
            # Start pickle file
            file = self.set_up_pickle(direction, hand_name, ratios, "live", trial)
            pkl.dump(save_list,file)
            file.close()
            print("File saved.")
        except KeyboardInterrupt:
            # If we manually stop execution, stop thread and save if we want
            self.event.set()
            print("Stopping thread")
            sleep(1)
            # Start pickle file
            file = self.set_up_pickle(direction, hand_name, ratios, "live", trial)
            pkl.dump(save_list,file)
            file.close()
            print("File saved.")
            
    
    def move_thread(self, hand_name, actual_pose, contact_point_l, direction, contact_point_r, ik_left, ik_right, contact_delta_l, contact_delta_r, m0, m1, m2, m3, m4, m5):
        dy = threading.Thread(target=self.dyn_move, args=(hand_name, actual_pose, contact_point_l, direction, contact_point_r, ik_left, ik_right, contact_delta_l, contact_delta_r, m0, m1, m2, m3, m4, m5, ), daemon=True)
        dy.start()

    def dyn_move(self, hand, current_obj_pose, contact_point_l, direction, contact_point_r, ik_left, ik_right, contact_delta_l, contact_delta_r, m0, m1, m2, m3, m4, m5):
        limit = .5
        # Try lower step values, increase them as neccessary
        step = .02
        counter = 0
        while not self.move_complete:
            if self.event.is_set():
                return
            # Calculate the target point for the left finger
            #print(f"Step Size {step}")
            if step > .045:
                # Break
                #print("Step size exceeded .4")
                self.done = True
                return 
            shifted_by_start_l = [contact_point_l[0]-self.initial_pose[0], contact_point_l[1]-self.initial_pose[1]]
            shifted_by_palm_l = [shifted_by_start_l[0], shifted_by_start_l[1]+self.palm_shift]

            shifted_by_start_r = [contact_point_r[0]-self.initial_pose[0], contact_point_r[1]-self.initial_pose[1]]
            shifted_by_palm_r = [shifted_by_start_r[0], shifted_by_start_r[1]+self.palm_shift]
            # Take the object position and create vector towards goal 
            # Apply that vector the contact point
            
            # Vector from object to goal
            object_vec = [self.f2_direction_dict[direction][0]-current_obj_pose[0], self.f2_direction_dict[direction][1]-current_obj_pose[1]]
            object_vec_unit = object_vec/np.sqrt((object_vec[0]**2)+(object_vec[1]**2))
            object_vec_step = object_vec_unit*step
            l_point =  object_vec_step + shifted_by_palm_l
            r_point =  object_vec_step + shifted_by_palm_r
            """
            shifted_by_start_l = [contact_point_l[0]-self.initial_pose[0], contact_point_l[1]-self.initial_pose[1]]
            shifted_by_palm_l = [shifted_by_start_l[0], shifted_by_start_l[1]+self.palm_shift]
            l_targ = [shifted_by_palm_l[0]-.04, self.f2_direction_dict[direction][1]]
            l_point = self.step_towards_goal(shifted_by_palm_l, l_targ, step) # self.f2_direction_dict[direction]was .02 with smoothing
            
            # # Calculate the target point for the right finger
            shifted_by_start_r = [contact_point_r[0]-self.initial_pose[0], contact_point_r[1]-self.initial_pose[1]]
            shifted_by_palm_r = [shifted_by_start_r[0], shifted_by_start_r[1]+self.palm_shift]
            r_targ = [shifted_by_palm_r[0]-.04, self.f1_direction_dict[direction][1]]
            r_point = self.step_towards_goal(shifted_by_palm_r, r_targ, step) #self.f1_direction_dict[direction]
            """
            # Calculate the inverse kinematics for each finger
            #print("Joint_r_2", ik_right.finger_fk.current_angles)
            #print("IT NUM2",ik_right.finger_fk.current_angles)
            _, new_angles_l, num_itl = ik_left.calculate_ik(target = l_point, ee_location=[contact_delta_l[0], contact_delta_l[1], 1])
            _, new_angles_r, num_itr = ik_right.calculate_ik(target = r_point, ee_location=[contact_delta_r[0], contact_delta_r[1], 1])
            #print("IT NUM2",ik_right.finger_fk.current_angles)
            #print("IT NUM2",ik_right.finger_fk.link_rotations)
            #print("IT NUM", num_itl)
            #print("IT NUM2", num_itr)
            #print("IT NUM2",ik_right.finger_fk.calculate_forward_kinematics())
            #print("IT NUM2",ik_right.finger_fk.link_translations)
            #print("IT NUM2",ik_right.finger_fk.link_lengths)
            """
            Debug print statements
            print(F"Num L: {num_itl}, Num R: {num_itr}")
            print(f"FK: {ik_right.finger_fk.calculate_forward_kinematics()}")
            print(f"L: {contact_delta_l}, R: {contact_delta_r}")
            print(f"m0: {m0}, new_m0: {new_angles_r[0]}::: m1: {m1}, new_m1: {new_angles_r[1]}::: m2: {m2}, new_m2: {new_angles_l[0]}::: m3: {m3}, new_m3: {new_angles_l[1]}")
            print(f"Contact left: {shifted_by_palm_l}, Contact right: {shifted_by_palm_r}")
            print(f"Target left: {l_point}, Target Right: {r_point}")
            #print(joint_a)

            """
            print(f"Contact left: {shifted_by_palm_l}, Contact right: {shifted_by_palm_r}")
            print(f"Target left: {l_point}, Target Right: {r_point}")
            print(f"L: {contact_delta_l}, R: {contact_delta_r}")
            print(f"m0: {m0}, new_m0: {new_angles_r[0]}::: m1: {m1}, new_m1: {new_angles_r[1]}::: m2: {m2}, new_m2: {new_angles_l[0]}::: m3: {m3}, new_m3: {new_angles_l[1]}")#::: m4: {m4}, new_m4: {new_angles_l[1]}::: m5: {m5}, new_m5: {new_angles_l[2]}")

            limit = .6
            if hand == "3v3":
                if np.abs(new_angles_r[0]-m0) > limit or np.abs(new_angles_r[1]-m1) > limit or np.abs(new_angles_r[2]-m2) > limit or np.abs(new_angles_l[0]-m3) > limit or np.abs(new_angles_l[1]-m4) > limit or np.abs(new_angles_l[2]-m5) > limit:
                    print("Bad value")
                    counter += 1
                    if step < .001:
                        self.move_complete = True
                        break
                    else: 

                        step -= .005
                    if counter > 8:
                        self.move_complete = True
                        break
                    #self.move_complete = True
                    continue
            elif hand == "2v2":
                if np.abs(new_angles_r[0]-m0) > limit or np.abs(new_angles_r[1]-m1) > limit or np.abs(new_angles_l[0]-m2) > limit or np.abs(new_angles_l[1]-m3) > limit:
                    print("Bad value")
                    counter += 1
                    if step < .001:
                        self.move_complete = True
                        break
                    else: 

                        step -= .005
                    if counter > 8:
                        self.move_complete = True
                        break
                    #self.move_complete = True
                    continue
            
            # HERE WE START MOVEMENT thread
            # Update the goal
            self.block = True
            if m4 is None:
                # Then we have a 2v2
                print("ShouldS see this")
                goal0 = new_angles_r[0]
                goal1 = new_angles_r[1]
                goal2 = new_angles_l[0]
                goal3 = new_angles_l[1]
            elif m5 is None:
                # We have a 2v3
                print("Shouldnt see this")
                goal0 = new_angles_r[0]
                goal1 = new_angles_r[1]
                goal2 = new_angles_l[0]
                goal3 = new_angles_l[1]
                goal4 = new_angles_l[2]
                #self.dynamixel_control.update_goal(4, self.dynamixel_control.dxls[4].center_pos+self.dynamixel_control.convert_rad_to_pos(goal4))
            else:
                
                # We have a 3v3
                goal0 = new_angles_r[0]
                goal1 = new_angles_r[1]
                goal2 = new_angles_r[2]
                goal3 = new_angles_l[0]
                goal4 = new_angles_l[1]
                goal5 = new_angles_l[2]
                #self.dynamixel_control.update_goal(4, self.dynamixel_control.dxls[4].center_pos+self.dynamixel_control.convert_rad_to_pos(goal4))
                #self.dynamixel_control.update_goal(5, self.dynamixel_control.dxls[5].center_pos+self.dynamixel_control.convert_rad_to_pos(goal5))
            # Update all the positions with the following: center_position + difference in 0-1023 scale
            #num_points = 10
            #goal_0_array = np.
            '''
            max_rotation = .15 #rad
            if np.abs(goal0-m0) > max_rotation:
                if goal0>m0:
                    goal0 = m0+max_rotation
                else:
                    goal0 = m0-max_rotation
            if np.abs(goal1-m1) > max_rotation:
                if goal1>m1:
                    goal1 = m1+max_rotation
                else:
                    goal1 = m1-max_rotation
            if np.abs(goal2-m2) > max_rotation:
                if goal2>m2:
                    goal2 = m2+max_rotation
                else:
                    goal2 = m2-max_rotation
            if np.abs(goal3-m3) > max_rotation:
                if goal3>m3:
                    goal3 = m3+max_rotation
                else:
                    goal3 = m3-max_rotation
            if np.abs(goal4-m4) > max_rotation:
                if goal4>m4:
                    goal4 = m4+max_rotation
                else:
                    goal4 = m4-max_rotation
            if np.abs(goal5-m5) > max_rotation:
                if goal5>m5:
                    goal5 = m5+max_rotation
                else:
                    goal5 = m5-max_rotation
            '''
            num =10
            goal0_array = np.linspace(m0, goal0, num)
            goal1_array = np.linspace(m1, goal1, num)
            goal2_array = np.linspace(m2, goal2, num)
            goal3_array = np.linspace(m3, goal3, num)
            #goal4_array = np.linspace(m4, goal4, num)
            #goal5_array = np.linspace(m5, goal5, num)
            # goal0_array = np.linspace(m0, goal0, retstep=.01)
            # goal1_array = np.linspace(m1, goal1, retstep=.01)
            # goal2_array = np.linspace(m2, goal2, retstep=.01)
            # goal3_array = np.linspace(m3, goal3, retstep=.01)
            # goal4_array = np.linspace(m4, goal4, retstep=.01)
            # goal5_array = np.linspace(m5, goal5, retstep=.01)

            for i in range(num):
                self.dynamixel_control.update_goal(0, self.dynamixel_control.dxls[0].center_pos+self.dynamixel_control.convert_rad_to_pos(goal0_array[i]))
                self.dynamixel_control.update_goal(1, self.dynamixel_control.dxls[1].center_pos+self.dynamixel_control.convert_rad_to_pos(goal1_array[i]))
                self.dynamixel_control.update_goal(2, self.dynamixel_control.dxls[2].center_pos+self.dynamixel_control.convert_rad_to_pos(goal2_array[i]))
                self.dynamixel_control.update_goal(3, self.dynamixel_control.dxls[3].center_pos+self.dynamixel_control.convert_rad_to_pos(goal3_array[i]))
                #self.dynamixel_control.update_goal(4, self.dynamixel_control.dxls[4].center_pos+self.dynamixel_control.convert_rad_to_pos(goal4_array[i]))
                #self.dynamixel_control.update_goal(5, self.dynamixel_control.dxls[5].center_pos+self.dynamixel_control.convert_rad_to_pos(goal5_array[i]))
                self.dynamixel_control.send_goal()
                sleep(.005)
            #sleep(.35)
            self.block = False
            self.move_complete = True
            counter = 0
            # Read dynamixel position and wait until within 10% of goal
        


    
    def set_up_pickle(self, direction, hand_name, ratios, folder, trial=None):
        # Find if the folder for the hand exists, if not create it
        if trial:
            folder_path = os.path.join("/media/kyle/16ABA159083CA32B/kyle", folder, "trial_"+str(trial))
        else:
            folder_path = os.path.join("/media/kyle/16ABA159083CA32B/kyle", folder)
        path = os.path.abspath(folder_path)
        path_to = os.path.join(path, hand_name+"_"+ratios)
        folder_exist = os.path.exists(path_to)
        if not folder_exist:
            os.chdir(path)
            os.mkdir(hand_name+"_"+ratios)
        os.chdir(path_to)
        
        file_name = direction + "_" + hand_name + "_" + ratios + ".pkl"

        file_path = os.path.join(path_to, file_name)
        file_ex = os.path.isfile(file_path)

        if file_ex:
            user_in = input("File already exists, do you want to overwrite?? y to overwrite, enter for no: ")
            if user_in == 'y':
                os.remove(file_path)
            else:
                sys.exit()
        else:
            user_in = input("Do you want to save the results? y for yes or enter for no: ")
            if not user_in == 'y':
                sys.exit()
        # Pickle name
        file = open(file_name, 'wb')
        return file

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



    def dyn_setup(self, hand_type = "2v3"):
        self.dynamixel_control = dynamixel_control.Dynamixel()

        if hand_type == "2v2":
            self.dynamixel_control.add_dynamixel(ID_number=0, calibration=[154, 506, 783], shift = 0)   # Right proximal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=1, calibration=[241, 580, 1000], shift = 0)   # Right distal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=2, calibration=[120, 417, 726], shift = 0) # Left proximal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=3, calibration=[21, 445, 780], shift = 0)   # Left distal (finger 2)
        elif hand_type == "2v3":
            self.dynamixel_control.add_dynamixel(ID_number=0, calibration=[214, 489, 764], shift = 20)#18)   # Right proximal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=1, calibration=[168, 512, 912], shift = 0)   # Right distal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=2, calibration=[306, 450, 840], shift = -20)#-20) # Left proximal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=3, calibration=[120, 521, 771], shift = 0)   # Left intermediate (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=4, calibration=[148, 507, 913], shift = 0)   # Left distal (finger 2)
        elif hand_type == "3v3":
            self.dynamixel_control.add_dynamixel(ID_number=0, calibration=[160, 495, 750], shift = 25)#was 24#18)   # Right proximal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=1, calibration=[415, 583, 913], shift = 0)   # Right distal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=2, calibration=[111, 475, 910], shift = 0)#20) # Left proximal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=3, calibration=[248, 514, 838], shift = -25)#-10)#-21   # Left intermediate (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=4, calibration=[183, 520, 740], shift = 0)   # Left distal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=5, calibration=[155, 573, 907], shift = 50)#15)   # Left distal (finger 2) #was 428
        else:
            #hand_type == "3v3":
            print("hand type not implemented")
            sys.exit()

        self.dynamixel_control.setup_all()
        
    def dyn_replay(self, direction = "N", hand_name = "2v3", ratios="1.1_1.1_1.1_1.1", delay = 0.0):
        """ Sets up and executes the dynamixel replay. Will get frames and positons, as well as start the replay thread.

        Args:
            direction (str): Direction to move the object
                (default is "N")
            hand_name (str): Hand name/overall configuration
                (default is "2v2")
            ratios (str): The ratios of the hand (finger/palm/links) 
                (default is "1.1_1.1_1.1_1.1")
            delay (float): Time is seconds between steps (sending next goal to Dynamixels)
                (default is 0.0)

        Returns:
            none
        """

        name = hand_name + "_" + ratios
        file_name = direction+"_"+name+".pkl"
        folder_name = os.path.join("Open_Loop_Data", name)
        
        # Ok, so we start by setting up the classes
        ## ARUCO
        at = aruco.Aruco_Track(self.aruco_params)
        self.dyn_setup(hand_type=hand_name)
        self.dynamixel_control.update_PID(85,40,45) # I term was 25
        self.dynamixel_control.update_speed(100)
        # Move Dynamixels to starting position
        self.dynamixel_control.go_to_initial_position(folder_name,file_name)
        sleep(3)
        self.dynamixel_control.update_speed(400)
        # Wait for user input to start
        input("Enter to start")

        # Start RealSense
        at.start_realsense()

        # Start pickle file
        first_time = True
        frame_counter = 0 
        save_list = []              
        try:
            while True:
                if self.done:
                    break

                # Get the color image and point data
                color_image, vtx = at.get_frame()

                if color_image is None or vtx is None:
                    # Check that we actually recived an image and points
                    print("No image - trying again.")
                    continue

                # Wait for the first predetermined number of frames before performing calculations
                if frame_counter < 20:
                    frame_counter+=1
                    continue

                # Get our current object pose in pixel coordinates
                current_pose, corners, _ = at.object_pose(color_image, vtx, True)
                if current_pose is None:#not current_pose.any():
                    print("Failed to determine pose, getting a new image.")
                    # If unable to determine a pose, continue
                    continue
                           
                # Convert from from pixel coordinates to m w/ depth data
                object_pose = self._pix_to_m(current_pose[0:2], vtx)
                if first_time:
                    # If this is the first frame we are capturing, save this as our intial position to use for relative calculations
                    first_time = False
                    self.initial_pose = [object_pose[0], object_pose[1], current_pose[2]]

                    
                    # Start the motors
                    self.dyn_replay_thread(dyn_file_location = folder_name,dyn_file_name=file_name, delay = delay)
                    continue

                
                # Get the current motor positions
                #self.dynamixel_control.bulk_read_pos()  # Read the current motor positions we do this in the dynamixel class
                m0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right bottom
                m1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 - right top
                m2 = self.dynamixel_control.dxls[2].read_position_m # Get the position of motor 2 - left bottom
                m3 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 3 - left  int
                m4 = None
                m5 = None
                
                if hand_name == "2v3":
                    m4 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 4 - left top
                elif hand_name == "3v3":
                    m4 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 4 - left top
                    m5 = self.dynamixel_control.dxls[5].read_position_m # Get the position of motor 4 - left top

                if False:
                    cv2.aruco.drawDetectedMarkers(color_image, corners)
                    cv2.imshow("hi",color_image)
                    cv2.waitKey(1)
                
                actual_pose = np.subtract([object_pose[0], object_pose[1], current_pose[2]], self.initial_pose)
                data_dict = {"obj_pos": [actual_pose[0], actual_pose[1], .05], "obj_or": actual_pose[2], "angles": {"joint_1": m0, "joint_2": m1, "joint_3": m2, "joint_4": m3, "joint 5": m4, "joint 6": m5}}
                save_list.append(data_dict)
                #print("looping")
                
            # Start pickle file
            file = self.set_up_pickle(direction, hand_name, ratios, folder="replay")
            pkl.dump(save_list,file)
            sleep(.5)
            file.close()
            print("File saved.")
        except KeyboardInterrupt:
            # If we manually stop execution, stop thread and save if we want
            self.dynamixel_control.event.set()
            print("Stopping thread")
            sleep(.5)
            # Start pickle file
            file = self.set_up_pickle(direction, hand_name, ratios, folder="replay")
            pkl.dump(save_list,file)
            sleep(.5)
            file.close()
            print("File saved.")
            

    def dyn_replay_thread(self, dyn_file_location="Open_Loop_Data", dyn_file_name="angles_E.pkl", delay = .005):
        """ Starts a thread that replays dynamixel positions from a pickle file.

        Args:
            dyn_file_location (str): Folder/file path to the location that the pickle file is stored
                (default is "Open_Loop_Data")
            dyn_file_name (str): Name of the pickle file to replay
                (default is "angles_E.pkl")
            delay (float): Time is seconds between steps (sending next goal to Dynamixels)
                (default is .005)

        Returns:
            none
        """
        # Wait a tiny bit before starting
        sleep(2)

        # Start the thread
        dy = threading.Thread(target=self.dynamixel_control.replay_pickle_data, args=(dyn_file_location,dyn_file_name,delay,), daemon=True)
        dy.start()

    def _pix_to_m(self, input, vtx):
        """ Converts from pixel locations to x, y based on the RealSense depth data

        Args:
            input (list): List in pixel coordinates (x,y)
            vtx (arrat): Multidiensional array relating pixel location to x, y, z in camera frame

        Returns:
            converted (list): A list in the same shape as the input list but now in meters
        """

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

    def test_contour_visualizer(self):
        """ A test function for live visualizing countour finding.

        Args:
            none

        Returns:
            none        
        """
        # Ok, so we start by setting up the classes
        # Ok, so we start by setting up the classes
        ## ARUCO TRACKING
        at = aruco.Aruco_Track(self.aruco_params)
        ## CONTOUR FINDING
        contour = ContourFind()
        ## CONTACT CALCULATIONS
        contact = ContactPoint(object_size=58.5)

        
           

        # Update values
        self.palm_shift = .16005
        ## DYNAMIXEL setup

        # Start RealSense
        at.start_realsense()    
        
        #os.path.dirname(__file__))
        #file_path = os.path.join(path_to, file_location, file_name)
    
        #with open(file_path, 'rb') as f:
        #    self.data = pkl.load(f)
        save_list = []

        first_time = True
        frame_counter = 0 
        while True:
            if self.done:
                break

            # Get the color image and point data
            color_image, vtx = at.get_frame()

            if color_image is None or vtx is None:
                # Check that we actually recived an image and points
                continue

            # Wait for the first predetermined number of frames before performing calculations
            if frame_counter < 40:
                frame_counter+=1
                continue

            # Get our current object pose in pixel coordinates
            current_pose, corners, ids = at.object_pose(color_image, vtx, True)
            print(ids)
            if current_pose is None:
                # If unable to determine a pose, continue
                continue
        
            # Get the contours back in pixel coordinates
            f_l_contour, f_r_contour, orig_c_left, orig_c_right = contour.find_countours(color_image)
            if f_l_contour is None:
                continue
            
            # Convert from from pixel coordinates to m w/ depth data
            object_pose = self._pix_to_m(current_pose[0:2], vtx)
            if first_time:
                first_time = False
                self.initial_pose = [object_pose[0], object_pose[1], current_pose[2]]
                continue
            
            finger_l_contour_m = self._pix_to_m(f_l_contour, vtx)
            finger_r_contour_m = self._pix_to_m(f_r_contour, vtx)
            zero_array = np.zeros(2)
            if np.all(np.isclose(finger_l_contour_m[0], zero_array)) or np.all(np.isclose(finger_l_contour_m[1], zero_array)) or np.all(np.isclose(finger_r_contour_m[0], zero_array)) or np.all(np.isclose(finger_r_contour_m[1], zero_array)):
                # We got a bad contour postion
                print('Bad contour position!')
                continue

            


            # Take the contours and object pose and calculate contact points
            contact_point_l, contact_delta_l = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_l_contour_m, [0,0], "L", dist_length=self.left_dist_length, sleeve_length=self.left_sleeve_length)
            contact_point_r, contact_delta_r = contact.contact_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_r_contour_m, [0,0], "R", dist_length=self.right_dist_length, sleeve_length=self.right_sleeve_length)

            #contact_delta_l[1] = min(contact_delta_l[1], .072)
            #contact_delta_r[1] = min(contact_delta_r[1], .072)

            actual_pose = np.subtract([object_pose[0], object_pose[1], current_pose[2]], self.initial_pose)
            #print("looping")

            show_image = True
            if show_image:
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
                # Draw contours        
                contour_image = cv2.drawContours(color_image, [orig_c_left, orig_c_right], -1, (0, 255, 255), 3)
                #cv2.aruco.drawDetectedMarkers(color_image, corners)
                
                # Draw a red circle with zero radius and -1 for filled circle
                image2 = cv2.circle(color_image, (int(current_pose[0]-x_l),int(current_pose[1]+y_l)), radius=5, color=(0, 0, 255), thickness=-1)
                image3 = cv2.circle(color_image, (int(current_pose[0]-x_r),int(current_pose[1]+y_r)), radius=5, color=(0, 0, 255), thickness=-1)

                cv2.imshow("hi", image3)
                #cv2.imwrite("points.png", image3)
                cv2.waitKey(10)



if __name__ == "__main__":
    manager = ik_manager()

    """
    pickle_files = ["angles_N.pkl", "angles_NE.pkl", "angles_E.pkl", "angles_SE.pkl", "angles_S.pkl", "angles_SW.pkl", "angles_W.pkl", "angles_NW.pkl"]
    for pkl in pickle_files:
        manager.linear_run(pkl)
    
    """
    manager.left_dist_length = .108
    manager.left_sleeve_length = .05
    manager.right_dist_length = .08
    manager.right_sleeve_length = .05
    #manager.test_contour_visualizer()
    manager.live_run(direction="N", hand_name="2v2", ratios="50.50_50.50_1.1_63", trial="3")
    
    # Uncomment this for live
    """
    manager.left_dist_length = .108
    manager.left_sleeve_length = .050
    manager.right_dist_length = .108
    manager.right_sleeve_length = .050

   
    """
    #manager.dyn_replay(direction="NW", hand_name="2v3", ratios="25.75_25.35.40_1.1_53")
    #manager.test_contour_visualizer()