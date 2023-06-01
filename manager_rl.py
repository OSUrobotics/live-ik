#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
THIS IS FOR NEW HAND
"""

import aruco_tracking as aruco
import contact_calculation 
from contour_find import ContourFind
import cv2
import numpy as np
import threading
import logging
import dynamixel_control.dynamixel_control as dynamixel_control
from time import time, sleep
import sys
from contact_calculation import ContactPoint
import pickle as pkl
import os
from copy import deepcopy
import importlib  
from DDPG_RW import DDPGfD_RW
import json
from StateRW import State_RW
hand = importlib.import_module("hand-gen-IK")
import csv
import pandas as pd

class ik_manager:

    def __init__(self, filepath):
        
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
            "N": np.array([0.0, .15]), # done
            "NE": np.array([0.15, .15]),
            "E": np.array([0.15, 0.0]),
            "SE": np.array([0.15, -.15]),
            "S": np.array([0.0, -.15]),
            "SW": np.array([-0.15, -.15]),
            "W": np.array([-0.15, 0.0]),
            "NW": np.array([-.15, .15])}
        self.f2_direction_dict = {
            "N": np.array([0.0, .15]), # done
            "NE": np.array([0.15, .15]),
            "E": np.array([0.15, 0.00]),
            "SE": np.array([0.15, -.15]),
            "S": np.array([0.0, -.15]),
            "SW": np.array([-0.15, -.15]),
            "W": np.array([-0.15, 0.0]),
            "NW": np.array([-.15, .15])}
        
        self.goal = [0,0]
        self.direction = "N"
            # Store the distace of the object from the palm (in y)
        self.palm_shift = .1#.1067*1.5 # .1 m from object to palm

        # Set parameters
        self.left_dist_length = .072
        self.right_dist_length = .072
        self.left_sleeve_length = .050
        self.right_sleeve_length = .050

        self.move_complete = True
        self.done = False
        self.block = False # Blocking variable to prevent reading at the same time as writing
        self.event = threading.Event()
        with open(filepath, 'r') as argfile:
            args = json.load(argfile)
        self.policy = DDPGfD_RW(args)
        self.useRL = True
        self.state = State_RW()


        self.joint_ang_RL = True

        self.csvreader = []
        # Test spoon feed the actor
        # Read in CSV and maintain the counter
        self.csvreader = pd.read_csv('actor_output.csv', sep=',', header=None)
        print(self.csvreader.values[0])
        print("yo")

        self.output_state = []
        self.output_actor = []

        self.csv_row = 0
    
    def load_policy(self, filepath):
        self.policy.load(filepath)

    def track_object_pos(self, direction = "NW", hand_name = "2v2", ratios="1.1_1.1_1.1_1.1", trial="1"):
        # Ok, so we start by setting up the classes
        ## ARUCO TRACKING
        at = aruco.Aruco_Track(self.aruco_params)
        frame_counter = 0 
        first_time = True
        fil_name= hand_name + "_" + ratios
        load_name = direction+"_"+fil_name+".pkl"
        file_name = direction+"_"+fil_name+"_"+trial+".pkl"
        folder_name = os.path.join("Open_Loop_Data", fil_name)
        # Start RealSense
        at.start_realsense()    
        self.dyn_setup(hand_type=hand_name)
        self.dynamixel_control.update_PID(1000,400,200) # I term was 25
        self.dynamixel_control.set_speed(100)
        # Move Dynamixels to starting position
        self.dynamixel_control.go_to_initial_position(folder_name, load_name)
        sleep(1)
        self.dynamixel_control.set_speed(30)
        
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
            palm_pose, current_pose, _, _ = at.aruco_poses(color_image, vtx, True)
            if current_pose is None:
                print("None")
                # If unable to determine a pose, continue
                continue
            
            # Convert from from pixel coordinates to m w/ depth data
            object_pose = self._pix_to_m(current_pose[0:2], vtx)
            palm_actual_pose = self._pix_to_m(palm_pose[0:2], vtx)

            #print("PALM ", palm_pose)
            #print("OBJECT ", current_pose)
            if first_time:
                # If this is the first frame we are capturing, save this as our intial position to use for relative calculations

                first_time = False
                obj_multiply = np.array([[object_pose[0]],[object_pose[1]],[1]])
                translated_pose = np.array([[1, 0, -palm_actual_pose[0]],[0, 1, -palm_actual_pose[1]]])@obj_multiply
                print("Translated pose", translated_pose)
                print("ANGLE ", current_pose[2])
                rotated_pose = np.array([[np.cos(-palm_pose[2]), -np.sin(-palm_pose[2])], [np.sin(-palm_pose[2]), np.cos(-palm_pose[2])]])@translated_pose
                print("Rotated pose: ", rotated_pose)
                self.initial_pose = [rotated_pose[0][0], rotated_pose[1][0], current_pose[2]-palm_pose[2]]
                print("Intial pose", self.initial_pose)
                #self.initial_pose = [object_pose[0], object_pose[1], current_pose[2]]
                continue       

            # We just need actual pose to reflect transformation/rotation
            obj_multiply = np.array([[object_pose[0]],[object_pose[1]],[1]])
            translated_pose = np.array([[1, 0, -palm_actual_pose[0]],[0, 1, -palm_actual_pose[1]]])@obj_multiply
            rotated_pose = np.array([[np.cos(-palm_pose[2]), -np.sin(-palm_pose[2])], [np.sin(-palm_pose[2]), np.cos(-palm_pose[2])]])@translated_pose
            shifted_to_actual_base = np.array([[1, 0, 0],[0, 1, -.033]])@[[rotated_pose[0][0]],[rotated_pose[1][0]],[1]]
            actual_pose = [shifted_to_actual_base[0][0], shifted_to_actual_base[1][0], current_pose[2]-palm_pose[2]]

            print("ACTUAL POSE ", actual_pose)
            #contact_delta_l[1] = min(contact_delta_l[1], .072)
            #contact_delta_r[1] = min(contact_delta_r[1], .072)


    
    def live_run(self, direction = "NW", hand_name = "2v2", ratios="1.1_1.1_1.1_1.1", trial="1"):
        # Ok, so we start by setting up the classes
        ## ARUCO TRACKING
        at = aruco.Aruco_Track(self.aruco_params)
        ## CONTOUR FINDING
        contour = ContourFind()
        ## CONTACT CALCULATIONS
        contact = ContactPoint(object_size=39) 

        
        fil_name= hand_name + "_" + ratios
        load_name = direction+"_"+fil_name+".pkl"
        file_name = direction+"_"+fil_name+"_"+trial+".pkl"
        folder_name = os.path.join("Open_Loop_Data", fil_name)
        #folder_name_init = os.path.join("Open_Loop_Data", name)

        # INVERSE KINEMATICS
        if hand_name == "2v2":
            
            testhand = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]], "offset": [.0275, 0, 0]},
                "finger2": {"name": "finger1", "num_links": 2, "link_lengths": [[0, .072, 0], [0, .072, 0]], "offset": [-.0275, 0, 0]}}
        elif hand_name == "2v3":
            testhand = {"finger1": {"name": "finger0", "num_links": 2, "link_lengths": [[0, .054, 0], [0, .162, 0]], "offset": [.04, 0, 0]},
                "finger2": {"name": "finger1", "num_links": 3, "link_lengths": [[0, .054, 0], [0, .0756, 0], [0, .0864, 0]], "offset": [-.04, 0, 0]}}
        elif hand_name == "3v3":
            testhand = {"finger1": {"name": "finger0", "num_links": 3, "link_lengths": [[0, .108, 0], [0, .054, 0], [0, .054, 0]], "offset": [.047625, 0, 0]},
                "finger2": {"name": "finger1", "num_links": 3, "link_lengths": [[0, .054, 0], [0, .0972, 0], [0, .06479, 0]], "offset": [-.047625, 0, 0]}}
        ik_left = hand.liveik.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger2"])
        ik_right = hand.liveik.JacobianIKLIVE(hand_id=1, finger_info=testhand["finger1"])

        # Update values
        #self.palm_shift = .16005
        ## DYNAMIXEL setup
        self.dyn_setup(hand_type=hand_name)
        self.dynamixel_control.update_PID(1000,400,200) # I term was 25
        self.dynamixel_control.set_speed(100)
        # Move Dynamixels to starting position
        #self.dynamixel_control.go_to_initial_position(folder_name, load_name)
        sleep(1)
        self.dynamixel_control.set_speed(30)
        # Wait for user input to start
        input("enter to continue")

        # Start RealSense
        at.start_realsense()    
        self.state.set_goal_pose(self.goal)
        
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
                palm_pose, current_pose, _, _ = at.aruco_poses(color_image, vtx, True)
                if current_pose is None:
                    print("None")
                    # If unable to determine a pose, continue
                    continue
                
                # Convert from from pixel coordinates to m w/ depth data
                object_pose = self._pix_to_m(current_pose[0:2], vtx)
                palm_actual_pose = self._pix_to_m(palm_pose[0:2], vtx)

                #print("PALM ", palm_pose)
                #print("OBJECT ", current_pose)
                if first_time:
                    # If this is the first frame we are capturing, save this as our intial position to use for relative calculations
                    self.dynamixel_control.bulk_read_pos()   # Read the current motor positions
                    first_time = False
                    obj_multiply = np.array([[object_pose[0]],[object_pose[1]],[1]])
                    translated_pose = np.array([[1, 0, -palm_actual_pose[0]],[0, 1, -palm_actual_pose[1]]])@obj_multiply
                    print("Translated pose", translated_pose)
                    print("ANGLE ", current_pose[2])
                    rotated_pose = np.array([[np.cos(-palm_pose[2]), -np.sin(-palm_pose[2])], [np.sin(-palm_pose[2]), np.cos(-palm_pose[2])]])@translated_pose
                    print("Rotated pose: ", rotated_pose)
                    self.initial_pose = [rotated_pose[0][0], rotated_pose[1][0], current_pose[2]-palm_pose[2]]
                    print("Intial pose", self.initial_pose)
                    #self.initial_pose = [object_pose[0], object_pose[1], current_pose[2]]
                    continue
                
                # Get the current motor positions 
                #self.dynamixel_control.bulk_read_pos()            
                m0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right prox
                m1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 
                m2 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 2 
                m3 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 3
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


                joint_right = deepcopy([m0, m1])
                joint_left = deepcopy([m2, m3])

                ik_left.update_angles(deepcopy(joint_left))
                ik_right.update_angles(deepcopy(joint_right))
                ik_left.finger_fk.update_ee_end_point([0, .072, 1])
                ik_right.finger_fk.update_ee_end_point([0, .072, 1])

                left = ik_left.finger_fk.calculate_forward_kinematics()
                right = ik_right.finger_fk.calculate_forward_kinematics()

                left_mid = ik_left.finger_fk.calculate_forward_kinematics_mid()
                right_mid = ik_right.finger_fk.calculate_forward_kinematics_mid()


                fingertip_pose = [right[0], right[1], left[0], left[1]]
                mid_pose = [right_mid[0], right_mid[1], left_mid[0], left_mid[1]]


                # We just need actual pose to reflect transformation/rotation
                obj_multiply = np.array([[object_pose[0]],[object_pose[1]],[1]])
                translated_pose = np.array([[1, 0, -palm_actual_pose[0]],[0, 1, -palm_actual_pose[1]]])@obj_multiply
                rotated_pose = np.array([[np.cos(-palm_pose[2]), -np.sin(-palm_pose[2])], [np.sin(-palm_pose[2]), np.cos(-palm_pose[2])]])@translated_pose
                print("ROTS ", rotated_pose)
                shifted_to_actual_base = np.array([[1, 0, 0],[0, 1, -.033]])@[[rotated_pose[0][0]],[rotated_pose[1][0]],[1]]
                actual_pose = [shifted_to_actual_base[0][0], shifted_to_actual_base[1][0], current_pose[2]-palm_pose[2]]

                print("ACTUAL POSE ", actual_pose)
                #contact_delta_l[1] = min(contact_delta_l[1], .072)
                #contact_delta_r[1] = min(contact_delta_r[1], .072)

                #actual_pose = np.subtract([object_pose[0], object_pose[1], current_pose[2]], self.initial_pose)
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

                    cv2.imshow("hi", color_image)
                    cv2.waitKey(5)

                if self.move_complete and not self.done:
                    # Update our angles in the FK with current motor angles
                    self.dynamixel_control.bulk_read_pos()
                    m0 = self.dynamixel_control.dxls[0].read_position_m # Get the position of motor 0 - right prox
                    m1 = self.dynamixel_control.dxls[1].read_position_m # Get the position of motor 1 
                    m2 = self.dynamixel_control.dxls[3].read_position_m # Get the position of motor 2 
                    m3 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 3
                    #m4 = self.dynamixel_control.dxls[4].read_position_m # Get the position of motor 4 - left intermediate
                    #m5 = self.dynamixel_control.dxls[5].read_position_m # Get the position of motor 5 - left distal
                    joint_right = deepcopy([m0, m1])
                    joint_left = deepcopy([m2, m3])

                    ik_left.update_angles(deepcopy(joint_left))
                    ik_right.update_angles(deepcopy(joint_right))
                    #print("Joint_r_1", ik_right.finger_fk.current_angles)
                    #print("starting thread")
                    self.move_complete = False
                    #self.block = True
                    print("Starting thread!!!!!!!!!")
                    self.move_thread(hand_name, actual_pose, direction, ik_left, ik_right, m0, m1, m2, m3, m4, m5, fingertip_pose, mid_pose)
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
            with open(self.direction+"_JA_state.pkl","wb") as file:

                
                pkl.dump(self.output_state,file)

            with open(self.direction+"_JA_actor.pkl","wb") as file:

                
                pkl.dump(self.output_actor,file)

            file = self.set_up_pickle(direction, hand_name, ratios, "live", trial)
            pkl.dump(save_list,file)
            file.close()
            print("File saved.")
            self.dynamixel_control.end_program()
            
    
    def move_thread(self, hand_name, actual_pose, direction, ik_left, ik_right, m0, m1, m2, m3, m4, m5,  fingertip_pose, mid_pose):
        dy = threading.Thread(target=self.dyn_move, args=(hand_name, actual_pose, direction, ik_left, ik_right, m0, m1, m2, m3, m4, m5,  fingertip_pose, mid_pose, ), daemon=True)
        dy.start()

    def dyn_move(self, hand, current_obj_pose, direction, ik_left, ik_right, m0, m1, m2, m3, m4, m5,  fingertip_pose, mid_pose):
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
                        # Take the object position and create vector towards goal 
            # Apply that vector the contact point
            
            if not self.useRL:
                # Vector from object to goal
                pass
            else:
                # NEED - object pose, fingertip pose, fingerbase pose and joint angles
                # TODO - check that initial units on these are correct
                # TODO - check that normalization of system is correct
                """ee_right = ik_right.finger_fk.calculate_forward_kinematics()
                ee_left = ik_left.finger_fk.calculate_forward_kinematics()
                print(f"Fingertip left {ee_left}, right {ee_right}")
                mid_right = ik_right.finger_fk.calculate_forward_kinematics_mid()
                mid_left = ik_left.finger_fk.calculate_forward_kinematics_mid()
                print(f"Fingertip left {mid_left}, right {mid_right}")
                fingerbase_pose = [mid_left[0], mid_left[1], mid_right[0], mid_right[1]]
                fingertip_pos = [ee_left[0], ee_left[1], ee_right[0], ee_right[1]]"""
                #print(f"Fingertip points {fingertip_pose}")
                #print(f"Mid pose {mid_pose}")


                joint_angles = {'l_prox_pin': m2,'l_distal_pin': m3,'r_prox_pin': m0, 'r_distal_pin': m1}
                shifted_obj_pose = current_obj_pose #[#current_obj_pose[0],current_obj_pose[1]+self.palm_shift,current_obj_pose[2]]
                #print("CURRENT POSE ", current_obj_pose)
                self.state.update_state(deepcopy(shifted_obj_pose), deepcopy(fingertip_pose), deepcopy(mid_pose), deepcopy(joint_angles))
                
                # gets a normalized actor output between -1 and 1
                actor_output = self.policy.select_action(self.state.get_state())
                #print(f"State: {self.state.get_state()['current_state']}")
                #print(f"Actor: {actor_output}")
                #actor_output = self.csvreader.values[self.csv_row]
                #print("COUNTER ", self.csv_row)
                #self.csv_row += 1
                #print(f"Actor 2: {actor_output}")
                
                self.output_state.append(deepcopy(self.state.get_state()))
                self.output_actor.append(deepcopy(actor_output))


                

                if not self.joint_ang_RL:
                    l_point = [0.002*actor_output[2] + fingertip_pose[2],0.002*actor_output[3] + fingertip_pose[3]]
                    r_point = [0.002*actor_output[0] + fingertip_pose[0],0.002*actor_output[1] + fingertip_pose[1]]
                    print(f"Left target: {l_point}, right target: {r_point}")
                    
                    _, new_angles_l, num_itl = ik_left.calculate_ik(target = l_point, ee_location=[0, .072, 1])
                    _, new_angles_r, num_itr = ik_right.calculate_ik(target = r_point, ee_location=[0, .072, 1])
                else:
                    new_angles_l = [joint_angles['l_prox_pin'] + 0.01*actor_output[2],joint_angles['l_distal_pin'] + 0.01*actor_output[3]]
                    new_angles_r = [joint_angles['r_prox_pin'] + 0.01*actor_output[0],joint_angles['r_distal_pin'] + 0.01*actor_output[1]]
                    num_itl, num_itr = 1,1
         

            print(f"m0: {m0}, new_m0: {new_angles_r[0]}::: m1: {m1}, new_m1: {new_angles_r[1]}::: m2: {m2}, new_m2: {new_angles_l[0]}::: m3: {m3}, new_m3: {new_angles_l[1]}")        
                
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
            #print(f"Contact left: {shifted_by_palm_l}, Contact right: {shifted_by_palm_r}")
            #print(f"Target left: {l_point}, Target Right: {r_point}")
            #Sprint(f"L: {contact_delta_l}, R: {contact_delta_r}")
            #print(f"m0: {m0}, new_m0: {new_angles_r[0]}::: m1: {m1}, new_m1: {new_angles_r[1]}::: m2: {m2}, new_m2: {new_angles_l[0]}::: m3: {m3}, new_m3: {new_angles_l[1]}")

            limit = 1#.6
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
                goal0 = new_angles_r[0]
                goal1 = new_angles_r[1]
                goal2 = new_angles_l[0]
                goal3 = new_angles_l[1]
            elif m5 is None:
                # We have a 2v3
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

            self.dynamixel_control.update_goal(0, self.dynamixel_control.dxls[0].center_pos+self.dynamixel_control.convert_rad_to_pos(goal0))
            self.dynamixel_control.update_goal(1, self.dynamixel_control.dxls[1].center_pos+self.dynamixel_control.convert_rad_to_pos(goal1))
            self.dynamixel_control.update_goal(3, self.dynamixel_control.dxls[3].center_pos+self.dynamixel_control.convert_rad_to_pos(goal2))                
            self.dynamixel_control.update_goal(4, self.dynamixel_control.dxls[4].center_pos+self.dynamixel_control.convert_rad_to_pos(goal3))
            self.dynamixel_control.send_goal()
          
            sleep(.1)
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
            self.dynamixel_control.add_dynamixel(type="XL-330", ID_number=0, calibration=[1023, 2048, 3073], shift = 0)   # Right proximal (finger 1)
            self.dynamixel_control.add_dynamixel(type="XL-330", ID_number=1, calibration=[1023, 2048, 3073], shift = 0)   # Right distal (finger 1)
            self.dynamixel_control.add_dynamixel(type="XL-330", ID_number=3, calibration=[1023, 2048, 3073], shift = 0) # Left proximal (finger 2)
            self.dynamixel_control.add_dynamixel(type="XL-330", ID_number=4, calibration=[1023, 2048, 3073], shift = 0)   # Left distal (finger 2)
        elif hand_type == "2v3":
            self.dynamixel_control.add_dynamixel(ID_number=0, calibration=[214, 500, 764], shift = 0)#18)   # Right proximal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=1, calibration=[168, 479, 912], shift = 0)   # Right distal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=2, calibration=[306, 560, 840], shift = 0)#-20) # Left proximal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=3, calibration=[120, 437, 771], shift = 0)   # Left intermediate (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=4, calibration=[148, 585, 913], shift = 0)   # Left distal (finger 2)
        elif hand_type == "3v3":
            self.dynamixel_control.add_dynamixel(ID_number=0, calibration=[137, 431, 775], shift = 10)#was 24#18)   # Right proximal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=1, calibration=[377, 573, 910], shift = 0)   # Right distal (finger 1)
            self.dynamixel_control.add_dynamixel(ID_number=2, calibration=[138, 481, 857], shift = 20) # Left proximal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=3, calibration=[198, 511, 785], shift = -10)#-21   # Left intermediate (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=4, calibration=[120, 454, 820], shift = 0)   # Left distal (finger 2)
            self.dynamixel_control.add_dynamixel(ID_number=5, calibration=[124, 529, 880], shift = 15)   # Left distal (finger 2)
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
            #print(ids)
            if current_pose is None:
                # If unable to determine a pose, continue
                print("No pose")
                continue
        
            # Get the contours back in pixel coordinates
            f_l_contour, f_r_contour, orig_c_left, orig_c_right = contour.find_countours(color_image)
            if f_l_contour is None:
                print("No contour")
                continue
            
            # Convert from from pixel coordinates to m w/ depth data
            object_pose = self._pix_to_m(current_pose[0:2], vtx)
            if first_time:
                first_time = False
                self.initial_pose = [object_pose[0], object_pose[1], current_pose[2]]
                print("First time")
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

            # Run the endpoint calcs
            bottom_l, top_l = contact.joint_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_l_contour_m, [0,0], "L", dist_length=self.left_dist_length, sleeve_length=self.left_sleeve_length)
            bottom_r, top_r = contact.joint_point_calculation([object_pose[0], object_pose[1], current_pose[2]], finger_r_contour_m, [0,0], "R", dist_length=self.right_dist_length, sleeve_length=self.right_sleeve_length)

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
                # Now take a find the left contact point's number of pixels 
                x_l_t = int(10*(object_pose[0]-top_l[0])/diff_x) 
                y_l_t = int(10*(object_pose[1]-top_l[1])/diff_y)
                x_l_b = int(10*(object_pose[0]-bottom_l[0])/diff_x) 
                y_l_b = int(10*(object_pose[1]-bottom_l[1])/diff_y)

                # Now take a find the right contact point's number of pixels 
                x_r_t = int(10*(object_pose[0]-top_r[0])/diff_x) 
                y_r_t = int(10*(object_pose[1]-top_r[1])/diff_y)
                x_r_b = int(10*(object_pose[0]-bottom_r[0])/diff_x) 
                y_r_b = int(10*(object_pose[1]-bottom_r[1])/diff_y)
                # Draw contours        
                contour_image = cv2.drawContours(color_image, [orig_c_left, orig_c_right], -1, (0, 255, 255), 3)
                #cv2.aruco.drawDetectedMarkers(color_image, corners)
                
                # Draw a red circle with zero radius and -1 for filled circle
                image2 = cv2.circle(color_image, (int(current_pose[0]-x_l_t),int(current_pose[1]+y_l_t)), radius=5, color=(0, 0, 255), thickness=-1)
                image2 = cv2.circle(color_image, (int(current_pose[0]-x_l_b),int(current_pose[1]+y_l_b)), radius=5, color=(0, 0, 255), thickness=-1)
                image3 = cv2.circle(color_image, (int(current_pose[0]-x_r_t),int(current_pose[1]+y_r_t)), radius=5, color=(0, 0, 255), thickness=-1)
                image3 = cv2.circle(color_image, (int(current_pose[0]-x_r_b),int(current_pose[1]+y_r_b)), radius=5, color=(0, 0, 255), thickness=-1)

                cv2.imshow("hi", image3)
                cv2.imwrite("points.png", image3)
                cv2.waitKey(10)



if __name__ == "__main__":
    manager = ik_manager('RW_JA/experiment_config.json')
    manager.load_policy('RW_JA/policy')
    """
    pickle_files = ["angles_N.pkl", "angles_NE.pkl", "angles_E.pkl", "angles_SE.pkl", "angles_S.pkl", "angles_SW.pkl", "angles_W.pkl", "angles_NW.pkl"]
    for pkl in pickle_files:
        manager.linear_run(pkl)
    
    """
    manager.left_dist_length = .072
    manager.left_sleeve_length = .05
    manager.right_dist_length = .072
    manager.right_sleeve_length = .050


    manager.goal = [0.046, -.046]
    manager.direction = "SE"
    #manager.track_object_pos(direction="NW", hand_name="2v2", ratios="50.50_50.50_1.1_63", trial="3")

    manager.live_run(direction="NW", hand_name="2v2", ratios="50.50_50.50_1.1_63", trial="3")
    #manager.test_contour_visualizer()
    
    # Uncomment this for live
    """
    manager.left_dist_length = .108
    manager.left_sleeve_length = .050
    manager.right_dist_length = .108
    manager.right_sleeve_length = .050

    manager.live_run(direction="NW", hand_name="2v2", ratios="50.50_50.50_1.1_63", trial="3")
    """
    #manager.dyn_replay(direction="NW", hand_name="2v2", ratios="50.50_50.50_1.1_63")
    #manager.test_contour_visualizer()