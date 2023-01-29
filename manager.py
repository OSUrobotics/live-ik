import aruco_tracking as aruco
import cv2
import numpy as np
import threading
import logging
import dynamixel_control
class ik_manager:

    def __init__(self):
        pass

    def open_loop(self, dyn_file_location="Open_Loop_Data", dyn_file_name="angles_N.pkl"):
        ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                        "aruco_params": cv2.aruco.DetectorParameters_create(),
                        "marker_side_dims": 0.03,
                        "opencv_camera_calibration": np.array(((591.40261976, 0.0, 323.94871535),(0.0, 593.59306833, 220.0225822),(0.0, 0.0, 1.00000000))),
                        "opencv_radial_and_tangential_dists": np.array((0.07656341,  0.41328222, -0.02156859,  0.00270287, -1.64179927))
                        }
        at = aruco.Aruco_Track(ARUCO_PARAMS)

        # Set up Dynamixels
        self.dyn_replay_setup()
        self.Dynamixel_control.go_to_initial_position(dyn_file_location, dyn_file_name)

        at.start_realsense()
        #at.save_frames()
        try:
            at.start_save_frames_thread(save=False, save_delay=0.0, live = True, live_delay=0.0)
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

    def dyn_replay_setup(self):
        self.Dynamixel_control = dynamixel_control.Dynamixel()
        self.Dynamixel_control.add_dynamixel(self.Dynamixel_control.create_dynamixel_dict(ID_number=0, calibration=[0, 460, 1023]))
        self.Dynamixel_control.add_dynamixel(self.Dynamixel_control.create_dynamixel_dict(ID_number=1, calibration=[0, 450, 1023]))
        self.Dynamixel_control.add_dynamixel(self.Dynamixel_control.create_dynamixel_dict(ID_number=2, calibration=[0, 490, 1023]))
        self.Dynamixel_control.add_dynamixel(self.Dynamixel_control.create_dynamixel_dict(ID_number=3, calibration=[0, 515, 1023]))


        self.Dynamixel_control.setup_all()
        
    def dyn_replay(self, dyn_file_location, dyn_file_name):
        self.Dynamixel_control.replay_pickle_data(dyn_file_location, dyn_file_name, delay_between_steps = .01)


if __name__ == "__main__":
    manager = ik_manager()
    manager.open_loop()
