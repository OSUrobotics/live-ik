## Created by Kyle DuFrene
# Based on previous work by John Morrow
import cv2
import pyrealsense2 as rs
import os
from sys import exit as ex
import numpy as np
from time import time, sleep
import threading
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math


class Aruco_Track:
    """
    A class to capture, record, and track aruco markers and positions.

    ...

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, ARUCO_PARAMS):

        """
        """
        self.ARUCO_PARAMS = ARUCO_PARAMS

        self.first_trial = True
        self.first_tvec = []
        self.first_rvec = []
        self.first_corner = []

        self.current_pos = [0,0,0] #[x,y,rot] in meters and rad

        self.event = threading.Event()


    def start_realsense(self):
        """
        Starts RealSense pipeline
        """
        self.pipe = rs.pipeline()
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipe)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
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

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipe.start(self.config)


    def save_frames(self, save = False, save_delay = 0.0, live = True, live_delay = 0.0, drive="/media/kyle/Asterisk", folder="Data", hand = "blank", direction = 'N', trial = 1):
        """Get the color frames from the RealSense and save them to the the provided folder. To be used with post-processing, Asterisk analysis, and live IK.

        Args:
            save (bool): Whether or not to save frames on a drive for asterisk analysis
                (default is False)
            save_delay (float): Delay in seconds between saving frames. Increase to reduce the number of frames saved
                (default is 0.0)
            live (bool): Whether or not update temporary image to use for live IK 
                (default is True)
            live_delay (float): Delay in seconds between updating live frame. Increase to reduce the number of frames saved
                (default is 0.0)
            drive (string): Path to external hard drive for saving images
                (default is "/media/kyle/Asterisk")
            folder (string): Path/folder to location of saving Asterisk images on external drive
                (default is "Data")
            hand ("string"): Folder named for the current hand 
                (default is "blank")
            direction ("string"): Direction/rotation of the trial
                (default is "N")
            trial (int): Trial number for that hand and direction
                (default is 1)

        Returns:
            none
        """

        # TODO: Decide how to call image processing once this is updated??

        prev_save_time = time()
        prev_save_live_time = time()

        # Check/create the path to save the images 
        if save:
            save_path = drive

            # Check if an external drive is attached
            if not os.path.isdir(save_path):
                ex("Drive not attached!!")
            os.chdir(save_path)

            # Check if the Data folder already exists yet
            save_path = os.path.join(save_path, folder)
            if not os.path.isdir(save_path):
                os.mkdir(folder)
            os.chdir(save_path)

            # Check if the hand folder already exists
            save_path = os.path.join(save_path, hand)
            if not os.path.isdir(save_path):
                os.mkdir(hand)
            os.chdir(save_path)

            # Check if the direction folder already exists
            save_path = os.path.join(save_path, direction)
            if not os.path.isdir(save_path):
                os.mkdir(direction)
            os.chdir(save_path)

            # Check if the direction folder already exists
            save_path = os.path.join(save_path, str(trial))
            if os.path.isdir(save_path):
                if os.listdir(save_path) != 0:
                    ex("Trial folder already exists and has content, please check and try again!!")
            else:
                os.mkdir(str(trial))
            os.chdir(save_path)

        img_num = 0 # Counter for saving images

        try:
            while True:
                if self.event.is_set():
                    break
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                
                if save:
                    if (time() - prev_save_time) > save_delay:
                        # Save if enabled and enough time has elapsed
                        os.chdir(save_path)
                        file_name = "hand:"+str(hand)+"_dir:"+str(direction)+"_trial:"+str(trial)+"_frame:"+str(img_num)+".jpg"
                        img_num += 1
                        cv2.imwrite(file_name, color_image)
                        prev_save_time = time()
                
                if live:
                    if (time() - prev_save_live_time) > live_delay:
                        # Save if enabled and enough time has elapsed
                        # Update live image
                        self.live_thread(color_image)  # Start aruco analysis in another thread
                        prev_save_live_time = time()
                        #break
            
        finally:

            # Stop streaming
            print("Stopping RealSense pipeline")
            self.pipe.stop()
            print("RealSense pipeline stopped")

    def start_save_frames_thread(self, save = False, save_delay = 0.0, live = True, live_delay = 0.0, drive="/media/kyle/Asterisk", folder="Data", hand = "blank", direction = 'N', trial = 1):
        save_frame = threading.Thread(target=self.save_frames, args=(save, save_delay, live, live_delay, drive, folder, hand, direction, trial,), daemon=True)
        save_frame.start()


    def live_thread(self, color_image):
        """Starts the live_tracking_analysis function (finding Aruco markers) in another thread 

        Args:
            color_image (??): Color image from the Intel Realsense
            
        Returns:
            none
        """
        x = threading.Thread(target=self.live_tracking_analysis, args=(color_image,), daemon=True)
        x.start()


    def live_tracking_analysis(self, color_image):
        """Runs Aruco marker detection on one image. Updates the global position variables.

        Args:
            color_image (??): Color image from the Intel Realsense
            
        Returns:
            none
        """

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, self.ARUCO_PARAMS["aruco_dict"],
            parameters=self.ARUCO_PARAMS["aruco_params"])

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.ARUCO_PARAMS["marker_side_dims"], self.ARUCO_PARAMS["opencv_camera_calibration"], self.ARUCO_PARAMS["opencv_radial_and_tangential_dists"])

        if self.first_trial:
            if rvec is None:
                print("No Aruco marker in the first frame!!")
            else:
                # If the first run, save the starting position to use for relative calculations
                self.first_trial = False

                self.first_corner = corners
                self.first_rvec = rvec
                self.first_tvec = tvec
        else:
            # Save the relative position position
            self.current_pos = self.calc_poses(corners, rvec, tvec)
            #y = threading.Thread(target=self.fun_things, args=(), daemon=True)
            #y.start()
        

        

    #def fun_things(self):
    #    print("hi")
    #    sleep(1)
    #    print("ho")

    def live_plotting(self):
        fig, ax = plt.subplots(figsize=(15, 12))
        # set the axes limits
        ax.axis([-.2,.2,-.2,.2])
        ax.set_title("Relative Position in mm")
        # set equal aspect such that the circle is not shown as ellipse
        ax.set_aspect("equal")
        # create a point in the axes
        point, = ax.plot(0,0, marker=(4, 0, 0), markersize=20)

        # Updating function, to be repeatedly called by the animation
        def update(phi):
            # obtain point coordinates 
            # set point's coordinates
            print(self.current_pos)
            point.set_data([self.current_pos[0]],[self.current_pos[1]])
            point.set_marker((4, 0, math.degrees(self.current_pos[2])+45.0))
            point.set_markersize(50)
            return point,

        
        ani = animation.FuncAnimation(fig, update, interval=10)
 
        plt.show()

        # TODO: figure out these exceptions
        self.event.set()
    
    def live_plotting_thread(self):
        z = threading.Thread(target=self.live_plotting, args=(), daemon=True)
        z.start()


        

    def calc_poses(self, corners, next_rvec, next_tvec):
        """
        Returns calculates poses for the given ArucoCorner object
        """
        
        # pre-allocate the numpy array space
        data_len = 4
        pose_data = np.full((data_len, 3), np.nan)

        # get the initial pose (even if no relative pose, still need this for calculating the rotation magnitude) 
        #first_corners
        init_corners = self.first_corner
        init_rvec = self.first_rvec
        init_tvec = self.first_tvec
        # go through corners, calculate poses
        
        # if its all nans, just skip calculation
        try:
            if np.all(np.isnan(corners)):
                raise Exception("Row of nans, skipping calculation")

            rel_angle = self._angle_between(init_corners[0][0][0] - init_corners[0][0][2], corners[0][0][0] - corners[0][0][2])
            rel_rvec, rel_tvec = self._relative_position(init_rvec, init_tvec, next_rvec, next_tvec)

            # found the stack overflow for it?
            # https://stackoverflow.com/questions/51270649/aruco-marker-world-coordinates
            rotM = np.zeros(shape=(3, 3))
            cv2.Rodrigues(rel_rvec, rotM, jacobian=0)
            
            row_data = [-rel_tvec[0][0], -rel_tvec[1][0], -rel_rvec[2][0]]

        except Exception as e:
                row_data = [np.nan, np.nan, np.nan]

        pose_data = row_data

        return np.around(pose_data, decimals=4)

    def _relative_position(self, rvec1, tvec1, rvec2, tvec2):
        rvec1, tvec1 = np.expand_dims(rvec1.squeeze(),1), np.expand_dims(tvec1.squeeze(),1)
        rvec2, tvec2 = np.expand_dims(rvec2.squeeze(),1), np.expand_dims(tvec2.squeeze(),1)
        invRvec, invTvec = self._inverse_perspective(rvec2, tvec2)

        orgRvec, orgTvec = self._inverse_perspective(invRvec, invTvec)

        info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
        composedRvec, composedTvec = info[0], info[1]

        composedRvec = composedRvec.reshape((3, 1))
        composedTvec = composedTvec.reshape((3, 1))

        return composedRvec, composedTvec

    def _inverse_perspective(self, rvec, tvec):
        """
        found you! https://aliyasineser.medium.com/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
        """
        # print(rvec)
        # print(np.matrix(rvec[0]).T)
        R, _ = cv2.Rodrigues(rvec)
        R = np.matrix(R).T
        invTvec = np.dot(-R, np.matrix(tvec))
        invRvec, _ = cv2.Rodrigues(R)
        return invRvec, invTvec

    def _angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                example 1) angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                example 2) angle_between((1, 0, 0), (1, 0, 0))
                0.0
                example 3) angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
                *ahem* https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
                (look at highest voted answer, then scroll down to sgt_pepper and crizCraig's answer
        """
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)

        minor = np.linalg.det(
            np.stack((v1_u[-2:], v2_u[-2:]))
        )

        if minor == 0:
            sign = 1
        else:
            sign = -np.sign(minor)

        dot_p = np.dot(v1_u, v2_u)
        dot_p = min(max(dot_p, -1.0), 1.0)
        return sign * np.arccos(dot_p)

    def _unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)




if __name__ == "__main__":
    ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                    "aruco_params": cv2.aruco.DetectorParameters_create(),
                    "marker_side_dims": 0.03,
                    "opencv_camera_calibration": np.array(((591.40261976, 0.0, 323.94871535),(0.0, 593.59306833, 220.0225822),(0.0, 0.0, 1.00000000))),
                    "opencv_radial_and_tangential_dists": np.array((0.07656341,  0.41328222, -0.02156859,  0.00270287, -1.64179927))
                    }
    at = Aruco_Track(ARUCO_PARAMS)
    try:
        at.start_realsense()
        at.save_frames(save=False, save_delay=0.0, live = True, live_delay=1.5) # was 0
    finally: 
        at.event.set()
