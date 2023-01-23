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

'''
import subprocess
from curtsies import Input 
import roslaunch
import rospy
from time import sleep
import os



class ik_manager:

    def __init__(self, name):
        self.name = name

        # Change the path for the configuration file
        #os.environ["ROS_PYTHON_LOG_CONFIG_FILE"] = "/home/kyle/ik_paper/live-ik/python_loggin.conf"


    def launch_realsense(self):
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        cli_args3 = ['realsense2_camera', 'rs_camera.launch']
        roslaunch_file3 = roslaunch.rlutil.resolve_launch_arguments(cli_args3)
        self.parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file3,verbose=False)


        self.parent.start()

        started = False
        ## Wait for topic to start
        while not started:
            sleep(1)
            bro = rospy.get_published_topics()
            print(len(bro))
            print(bro)
            for i in range(len(bro)):
                if '/camera/color/image_raw' in bro[i]:
                    
                    print('yoooo')
                    print('yoooo')
                    print('yoooo')
                    print('yoooo')
                    print('yoooo')
                    print('yoooo')
                    print('yoooo')
                    started = True

                    break
            
        #self.parent.spin()

        #sleep(10)
        
        #process.stop()
        
        
        '''
        camera_cmd =  "roslaunch realsense2_camera rs_camera.launch" #"rosrun image_view image_saver image:=/camera/color/image_raw"
        a = subprocess.Popen("exec " + camera_cmd, shell=True)

        
        waiting = True
        with Input() as input_generator:
            for c in input_generator:
                while waiting:
                    print(c)

                    if c == '<SPACE>':
                        a.terminate()
                        print("KILLING CAMERA PROCESS")
                        waiting = False

                break
        '''

    def image_viewer(self):
        
        # Check that the RealSense process had started and is still running
        # try: 
        #     self.parent
        # except AttributeError:
        #     print("RealSense node never started! Skipping image_view.")
        #     return

        # If the RealSense process exists, start the camera viewer
        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        # cli_args1 = ['image_view', 'image_view', 'image:=/camera/color/image_raw']
        # roslaunch_file1 = roslaunch.rlutil.resolve_launch_arguments(cli_args1)
        # print("her")
        # self.viewer = roslaunch.parent.ROSLaunchParent(uuid, (roslaunch_file1, cli_args1[2:]))
        # self.viewer.start()
        # print("bro")
        # self.viewer.spin()
        # sleep(10)
        # self.viewer.shutdown()
        # print("made it here")

        #roslaunch_args1 = cli_args1[2:]
        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        # cli_args4 = ["image_view","image_view",'image:=/camera/color/image_raw']
        # roslaunch_args4 = cli_args4[2:]
        # roslaunch_file4 = roslaunch.rlutil.resolve_launch_arguments(cli_args4)[0]
        # launch_files = [(roslaunch_file4, roslaunch_args4)]
        # self.newed = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
        # print("here")
        # print("bro")
        # self.newed.start()
        # self.newed.spin()
        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)

        # cli_args = ["image_view","image_view"]
        # #roslaunch_args = cli_args[2:]
        # roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0])]#, roslaunch_args)]

        # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

        # parent.start()

        # package = 'image_view'
        # executable = 'image_view'
        # node = roslaunch.core.Node(package, executable,args='image:=/camera/color/image_raw')

        # launch = roslaunch.scriptapi.ROSLaunch()
        # launch.start()

        # process = launch.launch(node)
        # sleep(10)
        # print(process.is_alive())
        # process.stop()



        # node = roslaunch.core.Node('image_view','image_saver',args='image:=/camera/color/image_raw')
        # launch = roslaunch.scriptapi.ROSLaunch()
        # launch.start()

        # process = launch.launch(node)
        # print(process.is_alive())
        # sleep(10)
        # process.stop()


        camera_cmd = "rosrun image_view image_saver image:=/camera/color/image_raw"
        #rosrun image_view image_view image:=/camera/color/image_raw
        #roslaunch realsense2_camera rs_camera.launch
        print("   ")
        print("Ready to run camera...")
        print("   ")
        print("PRESS <SPACE> TO STOP CAMERA WHEN RUNNING")
        print("   ")
        input("Press <ENTER>, when ready, to start the camera")
        print("CAMERA STARTED")

        a = subprocess.Popen("exec " + camera_cmd, shell=True)

        waiting = True
        with Input() as input_generator:
            for c in input_generator:
                while waiting:
                    print(c)

                    if c == '<SPACE>':
                        a.terminate()
                        print("KILLING CAMERA PROCESS")
                        waiting = False

                break


    def end_process(self):
        """Kills RealSense ROS node"""

        # Check if Real Sense process exists and kill it
        try: 
            print("Killing RealSense process")
            self.parent.shutdown()
            print("Successfully killed RealSense process")
        except AttributeError:
            print("Success - RealSense process never started/doesn't exist")




if __name__ == "__main__":
    #  Create instance of the class
    ik = ik_manager("yo")

    # Run until ctr+c break
    try:
        # Launch RealSense node
        ik.launch_realsense()
        
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")
        print("here")

        ik.image_viewer()

    except KeyboardInterrupt:
        # End RealSense node
        ik.end_process()
    
    # End the RealSense node
    ik.end_process()




