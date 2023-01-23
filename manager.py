## Runs and controls the following:

''' 
All are classes 

1) object_tracking - tracks the object
Arguments:
aruco code

Track object position and rotation. Run in seperate thread???

2) dynamixel_control - controls the dynamixel xl-320 servos. 
Arguments: 
setup - number of servos/location/number

Methods:
get status
get position
send position

3) data_visulization
Provides live tracking of 

4) 


Considerations??
Smoothing - do we do one step at a time (possibly very slow and jittery), or do we generate a certain subset of moves and execute and verify?


Note that currently John's library uses 4x4_250 aruco dict
'''
import subprocess
from curtsies import Input 
from time import sleep, process_time
import os
import pyrealsense2 as rs
import numpy as np
import cv2
from aruco_tool.aruco_tool import ArucoFunc, show_image, PoseDetector
import threading
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class ik_manager:

    def __init__(self):
        self.ARUCO_PARAMS = {"aruco_dict": cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250), 
                             "aruco_params": cv2.aruco.DetectorParameters_create(),
                             "marker_side_dims": 0.03,
                             "opencv_camera_calibration": np.array(((591.40261976, 0.0, 323.94871535),(0.0, 593.59306833, 220.0225822),(0.0, 0.0, 1.00000000))),
                             "opencv_radial_and_tangential_dists": np.array((0.07656341,  0.41328222, -0.02156859,  0.00270287, -1.64179927))
                             }
        
        self.img_num = 0
        self.first_corner = []
        self.first_rvec =[]
        self.first_tvec =[]
        self.counter = 1
        self.first_trial = True
        self.current_pos = [0,0,0] #[x,y,rot] in meters and rad



        """
        Old from John's!!!
        self.opencv_camera_calibration = np.array(((617.0026849655, -0.153855356, 315.5900337131),  # fx, s,cx
                (0, 614.4461785395, 243.0005874753),  # 0,fy,cy
                (0, 0, 1)))
        self.opencv_radial_and_tangential_dists = np.array((0.1611730644, -0.3392379107, 0.0010744837, 0.000905697))
        """


        self.marker_side_dims = 0.03  # in meters

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

    
    def save_frames(self, directory="/home/kyle/ik_paper/live-ik/Images", direction = 'N', trial = 1, hand = 1):
        """
        Get the color frames from the RealSense and save them to the Images folder.
        
        """

        os.chdir(directory)
        self.img_num = 0

        try:
            while True:

                # Wait for a coherent pair of frames: depth and color
                frames = self.pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())

                color_colormap_dim = color_image.shape
                self.img_num += 1
                #file_name= "grid"+str(self.img_num)+".jpg"
                file_name = "3hand:"+str(hand)+"_dir:"+str(direction)+"_trial:"+str(trial)+"_frame:"+str(self.img_num)+".jpg"
                cv2.imwrite(file_name, color_image)
                print("hi")
                
                sleep(6)
                
        finally:

            # Stop streaming
            self.pipe.stop()

    
    def record(self):
        x = threading.Thread(target=self.save_frames(), args=(1,))
        x.start()


    def save_frame_overwrite(self):
        os.chdir("/home/kyle/ik_paper/live-ik/")
        # Loops until it gets a good color frame
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            file_name = "temp_img.jpg"
            cv2.imwrite(file_name, color_image)
            print("here")
            break

    def live_tracking_analysis(self, image="temp_img.jpg"):

        orig = cv2.imread(image)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(gray, self.ARUCO_PARAMS["aruco_dict"],
            parameters=self.ARUCO_PARAMS["aruco_params"])

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_side_dims, self.ARUCO_PARAMS["opencv_camera_calibration"], self.ARUCO_PARAMS["opencv_radial_and_tangential_dists"])

        if self.first_trial:
            # If the first run, save the starting position to use for relative calculations
            self.first_trial = False

            self.first_corner = corners
            self.first_rvec = rvec
            self.first_tvec = tvec
        else:
            #print(corners)
            self.current_pos = self.calc_poses(corners, rvec, tvec)

    def live_tracking(self):
        try:
            self.start_realsense()
            print("hi")
            self.tracking_start()
            self.tracking_test()
            print("ha")
        finally:
            # Stop streaming
            self.pipe.stop()
    
    def tracking(self):
        while True:
            self.save_frame_overwrite()
            self.live_tracking_analysis()
            #sleep(.1)



    def tracking_test(self):
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

        try:
            ani = animation.FuncAnimation(fig, update, interval=10)
        except:
            KeyboardInterrupt

        plt.show()

    def tracking_start(self):
        x = threading.Thread(target=self.tracking, args=(), daemon=True)
        x.start()


    
    
    def live_aruco(self, file, aruco_id = 0):
        """
        Calulates the aruco marker position for one frame
        """
        # see note above
        #file = "imj.jpg"
        #file = "Images/hand::1_dir:N_trial1_frame:"+str(self.img_num)+".jpg"   #imj.jpg" #str( Path(__file__).parent.resolve() / "file_location.jpg" )/home/kyle/ik_paper/live-ik/Images/hand:1_dir:N_trial:1_frame:0.jpg
        
        # run on a single image with one id to track 
        pose_array, self.ugh = self.af.single_image_analysis_single_id(file, aruco_id)
        #print(pose_array)
        #how_image(file, True, 3)

        return pose_array
            
    


    def image_viewer(self):
        pass


    def end_process(self):
        pass



    def calc_poses(self, corners, next_rvec, next_tvec):
        """
        Returns calculates poses for the given ArucoCorner object
        """
        # TODO: add modes so we can cycle between different implementations to what we need
        # pre-allocate the numpy array space
        data_len = 4

        pose_data = np.full((data_len, 8), np.nan)

        # get the initial pose (even if no relative pose, still need this for calculating the rotation magnitude) 
        #first_corners
        init_corners = self.first_corner
        #[init_rvec, init_tvec] = self._calc_single_pose(init_corners) # TODO: should we make it stricter, so that its guaranteed there are no np.nan?
        init_rvec = self.first_rvec
        init_tvec = self.first_tvec
        # go through corners, calculate poses
        
        # if its all nans, just skip calculation
        try:
            if np.all(np.isnan(corners)):
                print("bro")
                raise Exception("Row of nans, skipping calculation")

            #print(corners[0][0][0])
            rel_angle = self._angle_between(init_corners[0][0][0] - init_corners[0][0][2], corners[0][0][0] - corners[0][0][2])
            rel_rvec, rel_tvec = self._relative_position(init_rvec, init_tvec, next_rvec, next_tvec)
            
            translation_val = np.round(np.linalg.norm(rel_tvec), 4)
            rotation_val = rel_angle * 180 / np.pi

            # found the stack overflow for it?
            # https://stackoverflow.com/questions/51270649/aruco-marker-world-coordinates
            rotM = np.zeros(shape=(3, 3))
            cv2.Rodrigues(rel_rvec, rotM, jacobian=0)
            ypr = cv2.RQDecomp3x3(rotM)  # TODO: not sure what we did with this earlier... need to check

            #row_data = [rel_tvec[0][0], rel_tvec[1][0], rel_tvec[2][0], translation_val, rel_rvec[0][0], rel_rvec[1][0], rel_rvec[2][0], rotation_val]
            #row_data = [x,y,z,total trans, rot x, rot y, rot z, total rot]
            row_data = [-rel_tvec[0][0], -rel_tvec[1][0], -rel_rvec[2][0]]
            print(row_data)
        except Exception as e:
                row_data = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

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
    #  Create instance of the class
    ik = ik_manager()

    ik.live_tracking()
    #print("img 1")
    #ik.test_aruc("Images/1hand:1_dir:N_trial:1_frame:2.jpg",1)
    #print("img moved 6.5 cm")
    #ik.test_aruc("Images/1hand:1_dir:N_trial:1_frame:3.jpg",2)
    #print("img moved")
    #ik.test_aruc("Images/hand:1_dir:N_trial:1_frame:1.jpg",2)
    


    #ik.start_realsense()
    #ik.save_frames()
    #pose2 = ik.live_aruco("frame2.jpg",6)
    #print(pose2)
    #start = process_time()
    #pose3 = ik.live_aruco("frame3.jpg",6)
    #print(pose3)
    #print(process_time()-start)
    # print("First rot")
    # print(pose3[0][0][0])
    # print("tran x")
    # print(pose3[1][0][0])
    # print("tran y")
    # print(pose3[1][0][1])
    # pose4 = ik.live_aruco("frame4.jpg",6)



    # print(ik.ugh._relative_position(pose3[0],pose3[1],pose4[0],pose4[1]))

    #while True:
    #    pose = ik.live_aruco()
    #ik.live_aruco()

    # Run until ctr+c break
    #try:
        # Launch RealSense node
        #ik.launch_realsense()

        #ik.image_viewer()

    #except KeyboardInterrupt:
        # End RealSense node
        #ik.end_process()
    
    # End the RealSense node
    #ik.end_process()




    """
    Pose vectors
    pose[0][0][0] is rot
    pose[1][0][0] is x (positive towards W)
    pose[1][0][1] is y

    For testing:
    Frame 1 - 2 mostly just shift in x
    Frame 2-3 rotate 90 degrees

    Use aruco ID 6

    
    """




