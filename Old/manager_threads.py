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
class ik_manager:

    def __init__(self):
        
        # General camera parameters
        self.camera_calibration = np.array(((587.65822288, 0.0, 312.22279429),(0.0, 587.25425585, 242.52669574),(0.0, 0.0, 1.00000000)))
        self.r_t_dists = np.array((.0744065755, .144374443, -.000463894288, -.00363146720, -1.13198957))
        #old self.camera_calibration = np.array(((591.40261976, 0.0, 323.94871535),(0.0, 593.59306833, 220.0225822),(0.0, 0.0, 1.00000000)))
        #old self.r_t_dists = np.array((0.07656341,  0.41328222, -0.02156859,  0.00270287, -1.64179927))


        pass

    def get_user_params(self):
        user_in = input("Enter 'o' for Open Loop, 'c' for closed loop")
        if user_in == 'o':
            pass
        elif user_in == 'c':
            pass
        else:
            print('Invalid input, please try again!')

    def open_loop(self, dyn_file_location="Open_Loop_Data", dyn_file_name="angles_E.pkl"):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)

        # Set up Dynamixels
        self.dyn_replay_setup()
        self.dynamixel_control.go_to_initial_position(dyn_file_location, dyn_file_name)

        at.start_realsense()
        #at.save_frames()
        try:
            print("ho")
            at.start_save_frames_thread(save=False, save_delay=0.0, live = True, live_delay=0.0)

            counter = 0
            start_time = time()
            while at.first_trial == True:
                # Wait until we get a good reading of the aruco marker
                if time() - start_time > 5.0:
                    sys.exit("Cannot find aruco marker")
                pass


            print("Saved first aruco marker loc, can continue")
            at.live_plotting_thread()
            
            self.dyn_replay(dyn_file_location, dyn_file_name)
        finally:
            # Ending all threads
            print("Stopping all threads")
            at.event.set()
            main_thread = threading.current_thread()
            for t in threading.enumerate():
                if t is main_thread:
                    continue
                logging.debug('joining %s', t.getName())
                t.join()
            print("All threads joined")

    def contat_test(self):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)

        # Set up Dynamixels
        self.dyn_replay_setup()
        #self.dynamixel_control.go_to_initial_position(dyn_file_location, dyn_file_name)

        at.start_realsense()


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

    def save_one_image(self, file = "test.jpg"):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        at.start_realsense()
        at.save_one_image(file)

    def save_image_series(self, file = "test", delay = 1):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        at.start_realsense()
        sleep(4)
        i = 0
        time_prev = 0
        try:
            while True:
                im = at.get_image()
                if time() - time_prev > delay:
                    name = file+str(i)+".jpg"
                    cv2.imwrite(name, im)
                    time_prev = time()
                    i += 1
                    print("Image saved")
                cv2.imshow("window", im)
                cv2.waitKey(1)
        except KeyboardInterrupt:
            at.pipe.stop()

    def test_aruco(self):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)

        at.start_realsense()
        at.save_frames()


    def closed_loop(self, dyn_file_location="Open_Loop_Data", dyn_file_name="angles_E.pkl"):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": self.camera_calibration,
                        "opencv_radial_and_tangential_dists": self.r_t_dists
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)
        contact = ContactPoint()

        # Set up Dynamixels
        self.dyn_replay_setup()
        self.dynamixel_control.go_to_initial_position(dyn_file_location, dyn_file_name)
        

        at.start_realsense()
        
        sleep(5)
        

        try:
            # Get frames and get position of 
            at.start_save_frames_thread(save=False, save_delay=0.0, live = True, live_delay=0.08, contact = True)
            print("Start frame recording")
            # Calculate the object pos/loc
            counter = 0
            start_time = time()
            while at.first_trial == True:
                # Wait until we get a good reading of the aruco marker
                if time() - start_time > 5.0:
                    sys.exit("Cannot find aruco marker")
                pass
            


            print("Saved first aruco marker loc, can continue")
            at.live_plotting_thread()
            sleep(2)
            self.dyn_replay_thread(dyn_file_location, dyn_file_name, .01)
            
            while True:
                if at.updated:
                    at.updated = False
                    at.fing_1_contact = contact.contact_point_calculation(at.current_pos, at.finger_1)
                    at.fing_2_contact = contact.contact_point_calculation(at.current_pos, at.finger_2)
                    
                    
                    #ahh = cv2.aruco.drawDetectedMarkers(at.colo, at.corners, at.ids)
                    ahh = cv2.drawContours(at.colo, [at.c[0]], -1, (0, 255, 0), 3)
                    ahh = cv2.drawContours(ahh, [at.c[1]], -1, (0, 255, 0), 3)
                    cv2.imshow("hi",ahh)
                    cv2.waitKey(1)
            
            
            
            #self.dyn_replay(dyn_file_location, dyn_file_name)
        finally:
            # Ending all threads
            print("Stopping all threads")
            at.event.set()
            main_thread = threading.current_thread()
            for t in threading.enumerate():
                if t is main_thread:
                    continue
                logging.debug('joining %s', t.getName())
                t.join()
            print("All threads joined")

    def contat_test(self):
        pass

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
    pickle_files = ["angles_N.pkl", "angles_NE.pkl", "angles_E.pkl", "angles_SE.pkl", "angles_S.pkl", "angles_SW.pkl", "angles_W.pkl", "angles_NW.pkl"]
    for pkl in pickle_files:
        manager.linear_run(pkl)
    #manager.contour_visualizer()
    
    """
    dyn_file_location="Open_Loop_Data"
    dyn_file_name="angles_E.pkl"

    
    #manager.test_aruco()
    ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": manager.camera_calibration,
                        "opencv_radial_and_tangential_dists": manager.r_t_dists
                        }
    at = aruco.Aruco_Track(ARUCO_PARAMS)
    contact = ContactPoint()
    contour = ContourFind()

    # Set up Dynamixels
    manager.dyn_replay_setup()
    manager.dynamixel_control.go_to_initial_position(dyn_file_location, dyn_file_name)
    sleep(1)

    at.start_realsense()
    sleep(1)

    
    manager.dyn_replay_thread(dyn_file_location, dyn_file_name, .01)
    """



    #manager.save_image_series()
