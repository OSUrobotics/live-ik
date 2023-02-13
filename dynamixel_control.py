from dynamixel_sdk import *                    # Uses Dynamixel SDK library   
import dynamixel
from time import sleep
import os
import pickle as pkl
import numpy as np
from math import pi
from pathlib import Path

# TODO: tune PID https://www.youtube.com/watch?v=msWlMyx8Nrw&ab_channel=ROBOTISOpenSourceTeam

class Dynamixel:

    def __init__(self): 
        self.DEVICENAME = '/dev/ttyUSB3'
        self.PROTOCOL_VERSION = 2.0
        self.BAUDRATE = 57600

        self.portHandler = PortHandler(self.DEVICENAME)

        # Create flag for first bulk read
        self.first_bulk_read = True
        

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        # Initialize GroupBulkWrite instance
        self.groupBulkWrite = GroupBulkWrite(self.portHandler, self.packetHandler)

        # Initialize GroupBulkRead instace for Present Position
        self.groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

        # Create a dictionary of each Dynamixel object
        # key: id_number; value: Dxl object
        self.dxls = {}
            
    def create_dynamixel_dict(self, type = "XL-320", ID_number = 0, calibration = [0, 511, 1023], shift = 0) -> dict:
        """ Takes in parameters and creates a dictionary for the dynamixel.

        Args:
            type (string): Dynamixel type (model)
                (default is "XL-320")
            ID_number (int): Dynamixel ID number
                (default is 0)
            calibration (list of ints): List containing the minimum, center, and max positions of that joint (Dynamixel) 
                (default is [0, 511, 1023])
        Returns:
            dynamixel_dict (dict): Dictionary containing all of the parameters
        """
        dynamixel_dict = {"type": type,
                          "ID_number": ID_number,
                          "calibration": calibration,
                          "shift": shift}
        return dynamixel_dict

    def add_dynamixel(self, dyn_dict):
        """ Creates a Dxl bject and adds it to our dictionary based on the parameters passed in.

        Args:
            dyn_dict (dict): Dictionary of relavent Dynamixel settings
        Returns:
            none
        """
        
        # Get the ID number for easy reference
        id_num = dyn_dict["ID_number"]

        # Create a Dxl object and add it to our dictionary
        self.dxls[id_num] = dynamixel.Dxl(dyn_dict)

    def enable_torque(self, id: int, enable: bool = True):
        """ Enables or disables torque for one Dynamixel.

        Args:
            id (int): ID number of Dynamixel
            enable (boot): Enable torque (True) or disable torque (False)
                (default is True)
        Returns:
            none
        """

        # Choose enable or disable value
        if enable:
            enable_value = self.dxls[id].TORQUE_ENABLE
        else: 
            enable_value = self.dxls[id].TORQUE_DISABLE

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, id, self.dxls[id].ADDR_TORQUE_ENABLE, enable_value)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        elif enable:
            print("Dynamixel#%d has been successfully connected" % id)

    '''
    def setup_parameters(self, id: int, position = True, torque = True):
        """ Sets up the parameters to read position

        Args:
            id (int): ID number of Dynamixel
            position (bool): Whether to set up a position parameter
            torque (bool): Whether to set up a torque parameter
        Returns:
            none
        """
        if position:
            # Add parameter storage for Dynamixel present position
            dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxls[id].ADDR_PRESENT_POSITION, self.dxls[id].LEN_PRESENT_POSITION)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam failed" % id)
                quit()
    '''



    def setup_all(self):
        """ "Starts" all Dynamixels - this enables the torque and sets up the position parameter

        Args:
            none
        Returns:
            none
        """

        #  Enable torque for all Dyanmixels
        for id in self.dxls.keys():
            self.enable_torque(id, True)
        
        # Setup position parameter for all Dynamixels
        ##for id in self.dxls.keys():
         #   self.setup_parameters(id)

    def bulk_read_pos(self):
        """ Check and read current positions from each Dynamixel

        Args:
            none
        
        Returns:
            none
        
        """

        self.groupBulkRead.txRxPacket()

        # TODO: see if we need checks for each motor here??

        # Must set to 2 bytes otherwise errors!!
        if self.first_bulk_read: # TODO: THIS IS VERY SLOW TO ADD PARAMS, make sure it doesn't happen live
            for id in self.dxls.keys():
                # Add parameter storage for Dynamixel present position
                dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxls[id].ADDR_PRESENT_POSITION, self.dxls[id].LEN_PRESENT_POSITION)
                if dxl_addparam_result != True:
                    print("[ID:%03d] groupBulkRead addparam failed" % id)
                    quit()
            self.first_bulk_read = False

        self.groupBulkRead.txRxPacket()

        for id in self.dxls.keys():

            # Saves position read in each Dxl object
            self.dxls[id].read_position = self.groupBulkRead.getData(id, self.dxls[id].ADDR_PRESENT_POSITION, self.dxls[id].LEN_PRESENT_POSITION) - self.dxls[id].shift
            #print(f"Current pos: {self.dxls[id].read_position}")

        self.groupBulkRead.clearParam()
    
    def get_position(self, id: int):
        return self.dxls[id].read_position


    def bulk_read_torque(self):
        """ Check and read current positions from each Dynamixel

        Args:
            none
        
        Returns:
            none
        
        """

        # TODO: not sure if this is actually working...
        for id in self.dxls.keys():
            # Add parameter storage for Dynamixel present position
            dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxls[id].CURRENT_TORQUE_INDEX, self.dxls[id].LEN_CURRENT_TORQUE_INDEX)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam failed" % id)
                quit()
        
        self.groupBulkRead.txRxPacket()
        
        for id in self.dxls.keys():

            # Saves torque read in each object
            self.dxls[id].current_torque = self.groupBulkRead.getData(id, self.dxls[id].CURRENT_TORQUE_INDEX, self.dxls[id].LEN_CURRENT_TORQUE_INDEX)
            #print(f"Current torque: {self.dxls[id].current_torque}")
            
        self.groupBulkRead.clearParam()


    def send_goal(self):
        """ Writes goal positions to all Dynamixels based on goal position stored in each Dxl object.

        Args:
            none
        
        Returns:
            none
        
        """

        # Loop through all Dxls
        for id in self.dxls.keys():
            dxl = self.dxls[id]

            # Adjust goal by shift 
            goal = dxl.goal_position + dxl.shift

            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal)), DXL_HIBYTE(DXL_LOWORD(goal)), DXL_LOBYTE(DXL_HIWORD(goal)), DXL_HIBYTE(DXL_HIWORD(goal))]
            # Add Dynamixel goal position value to the Bulkwrite parameter storage
            dxl_addparam_result = self.groupBulkWrite.addParam(id, dxl.ADDR_GOAL_POSITION, dxl.LEN_GOAL_POSITION, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % id)
                quit()
        
        # Bulkwrite goal position and LED value
        # TODO: Add LED value stuff at some point if we want??
        dxl_comm_result = self.groupBulkWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear bulkwrite parameter storage
        self.groupBulkWrite.clearParam()
    
    def update_goal(self, id: int, new_goal: int):
        """ Updates the goal position stored in the object for 1 dynamixel

        Args:
            id (int): ID number of Dynamixel to update
            new_goal (int): New goal position from 0 to 1023
        Returns:
            none
        
        """
        self.dxls[id].goal_position = new_goal

        
    def end_program(self):
        """ Turns off Dynamixel torque and closes the port. Run this upone exit/program end.

        Args:
            none
        Returns:
            none
        
        """

        # Clear bulkread parameter storage
        self.groupBulkRead.clearParam()

        # Disable torque
        for id in self.dxls.keys():
            self.enable_torque(id, False)

        self.portHandler.closePort()  

    def load_pickle(self, file_location="Open_Loop_Data", file_name="angles_N.pkl") -> int:
        """ Open and load in the radian values (relative positions) from the pickle file. Convert them to positions from 0 to 1023 (how the Dynamixel XL-320 uses them). Updates the joint angles lists.

        Args:
            file_location (string): Path to folder where the pickle is saved
                (default is "/Open_Loop_Data")
            file_name (string): Name of pickle file
                (default is "angles_N.pkl")

        Returns:
            none
        """

        path_to = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(path_to, file_location, file_name)
    
        with open(file_path, 'rb') as f:
            data = pkl.load(f)

        for id in self.dxls.keys():
            name = "joint_" + str(id+1)
            self.dxls[id].joint_angles_pickle = self.convert_rad_to_pos(data[name])

        pickle_length = len(self.dxls[id].joint_angles_pickle)

        return pickle_length

    def convert_rad_to_pos(self, rad: float) -> int:
        """ Converts from radians to positions from 0 to 1023.

        Args:
            rad (float): Position value in radians
        Returns:
            pos (int): Position in range of 0 to 1023
        """
        
        # XL-320 is 1023 to 300 degrees
        # .2932 degrees per step
        # Convert to degrees
        deg = np.multiply(rad, (180/pi))
        
        # Pos is deviation from center (0 degrees), defined in init
        pos = np.multiply(deg, (1023/300))
        pos = pos.astype(int)

        return pos


    def map_pickle(self, i: int):
        """ Convert from relative to absolute positions based on the calibration. Updates global goal position variable.

        Args:
            i (int): Index of the list to convert
        Returns:
            none
        """

        # Set the positions in terms of actual calibrated motor positions
        for id in self.dxls.keys():
            self.dxls[id].goal_position = self.dxls[id].center_pos - self.dxls[id].joint_angles_pickle[i]


    def replay_pickle_data(self, file_location="Open_Loop_Data", file_name="angles_N.pkl", delay_between_steps: float = .01):
        
        try:    
            pickle_length = self.load_pickle(file_location, file_name)
            #self.map_pickle(0)
            #self.send_goal()
            #input("Press Enter to continue to next step.")
            for i in range(pickle_length):
                self.map_pickle(i)
                self.send_goal()
                sleep(delay_between_steps)
                self.bulk_read_pos()

        except KeyboardInterrupt:
            self.end_program()

    def go_to_initial_position(self, file_location="Open_Loop_Data", file_name="angles_N.pkl"):
        try: 
            pickle_length = self.load_pickle(file_location, file_name)
            self.map_pickle(0)
            self.send_goal()
        except:
            self.end_program()



if __name__ == "__main__":
    Dynamixel_control = Dynamixel()
    Dynamixel_control.add_dynamixel(Dynamixel_control.create_dynamixel_dict(ID_number=0, calibration=[0, 450, 1023], shift = -25)) # Negative on left side
    Dynamixel_control.add_dynamixel(Dynamixel_control.create_dynamixel_dict(ID_number=1, calibration=[0, 553, 1023], shift = 0))
    Dynamixel_control.add_dynamixel(Dynamixel_control.create_dynamixel_dict(ID_number=2, calibration=[0, 465, 1023], shift = 25)) # Positive on right side
    Dynamixel_control.add_dynamixel(Dynamixel_control.create_dynamixel_dict(ID_number=3, calibration=[0, 545, 1023], shift = 0))



    Dynamixel_control.setup_all()
    Dynamixel_control.replay_pickle_data(file_name="angles_E.pkl", delay_between_steps = .005)

    