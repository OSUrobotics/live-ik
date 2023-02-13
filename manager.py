import aruco_tracking as aruco
import contact_calculation 
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
        
    def dyn_replay(self, dyn_file_location, dyn_file_name, delay = .005):
        self.dynamixel_control.replay_pickle_data(dyn_file_location, dyn_file_name, delay_between_steps = delay)

    def dyn_replay_thread(self, dyn_file_location, dyn_file_name, delay = .005):
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
        sleep(2)

        at.start_realsense()
        try:
            # Get frames and get position of 
            at.start_save_frames_thread(save=False, save_delay=0.0, live = True, live_delay=0.0, contact = True)

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
            sleep(1)
            self.dyn_replay_thread(dyn_file_location, dyn_file_name, .01)
            
            while True:
                if at.updated:
                    print("woa")
                    at.updated = False
                    print(at.current_pos)
                    at.fing_1_contact = contact.contact_point_calculation(at.current_pos, at.finger_1)
                    at.fing_2_contact = contact.contact_point_calculation(at.current_pos, at.finger_2)
            
            
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



if __name__ == "__main__":
    manager = ik_manager()
    #manager.save_one_image()
    #manager.dyn_replay_setup()
    #manager.dyn_replay(dyn_file_location="Open_Loop_Data", dyn_file_name="angles_E.pkl", delay_between_steps = .005)
    
    
    #manager.test_aruco()
    manager.closed_loop()

    #manager.save_image_series()
