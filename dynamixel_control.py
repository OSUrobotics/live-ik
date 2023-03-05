from dynamixel_sdk import *                    # Uses Dynamixel SDK library   
import dynamixel
from time import sleep
import os
import pickle as pkl
import numpy as np
from math import pi
from pathlib import Path
import threading 

# TODO: tune PID https://www.youtube.com/watch?v=msWlMyx8Nrw&ab_channel=ROBOTISOpenSourceTeam

class Dynamixel:

    def __init__(self): 
        self.DEVICENAME = '/dev/ttyUSB0'
        self.PROTOCOL_VERSION = 2.0
        self.BAUDRATE = 57600

        self.portHandler = PortHandler(self.DEVICENAME)
        self.event = threading.Event()

        # Create flag for first bulk read
        self.first_bulk_read = True
        self.flag = False
        
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

    def add_dynamixel(self, type = "XL-320", ID_number = 0, calibration = [0, 511, 1023], shift = 0):
        """ Creates a Dxl bject and adds it to our dictionary based on the parameters passed in.

        Args:
            dyn_dict (dict): Dictionary of relavent Dynamixel settings
        Returns:
            none
        """
        dyn_dict = self.create_dynamixel_dict(type, ID_number, calibration, shift)
        
        # Create a Dxl object and add it to our dictionary
        self.dxls[ID_number] = dynamixel.Dxl(dyn_dict)

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

    def setup_all(self):
        """ "Starts" all Dynamixels - this enables the torque and sets up the position read parameter

        Args:
            none
        Returns:
            none
        """

        # Loop through the Dynamixels
        for id in self.dxls.keys():
            #  Enable torque for all Dyanmixels
            self.enable_torque(id, True)

            # Setup parameter to read dynamixel position
            # Add parameter storage for Dynamixel present position
            dxl_addparam_result = self.groupBulkRead.addParam(id, self.dxls[id].ADDR_PRESENT_POSITION, self.dxls[id].LEN_PRESENT_POSITION)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam failed" % id)
                quit()
        
        self.first_bulk_read = False
    

    def bulk_read_pos(self):
        """ Check and read current positions from each Dynamixel

        Args:
            none
        
        Returns:
            none        
        """

        # Read from the Dynamixels
        self.groupBulkRead.txRxPacket()
        # Must set to 2 bytes otherwise errors!!

        for id in self.dxls.keys():
            # Saves position read in each Dxl object
            self.dxls[id].read_position = self.groupBulkRead.getData(id, self.dxls[id].ADDR_PRESENT_POSITION, self.dxls[id].LEN_PRESENT_POSITION)
            if self.flag:
                self.dxls[id].read_position_m = self.convert_pos_to_rad(self.dxls[id].read_position - self.dxls[id].center_pos - self.dxls[id].shift)
            else:
                self.dxls[id].read_position_m = self.convert_pos_to_rad(self.dxls[id].read_position - self.dxls[id].center_pos)
  
    
    def get_position(self, id: int):
        return self.dxls[id].read_position

    
    def bulk_read_torque(self):
        """ NOT IS USE
        Check and read current positions from each Dynamixel

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

    def update_PID(self, P: int = 36, I: int = 0, D: int = 0):
        """ Updates the PID constants for all Dynamixels 

        Args:
            P (int): Proportional constant
            I (int): Integral constant
            D (int): Derivative constant
        
        Returns:
            none
        """

        # Loop through all Dxls and update the P value
        for id in self.dxls.keys():
            dxl = self.dxls[id]
            dxl_addparam_result = self.groupBulkWrite.addParam(id, dxl.ADDR_P, dxl.LEN_PID, [P])
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % id)
                print("wtf")
                quit()
        # Bulkwrite P values
        dxl_comm_result = self.groupBulkWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        # Clear bulkwrite parameter storage
        self.groupBulkWrite.clearParam()

        # Loop through all Dxls and update the I value
        for id in self.dxls.keys():
            dxl = self.dxls[id]
            dxl_addparam_result = self.groupBulkWrite.addParam(id, dxl.ADDR_I, dxl.LEN_PID, [I])
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % id)
                print("wtf2")
                quit()
        # Bulkwrite I values
        dxl_comm_result = self.groupBulkWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        # Clear bulkwrite parameter storage
        self.groupBulkWrite.clearParam()

        # Loop through all Dxls and update the D value
        for id in self.dxls.keys():
            dxl = self.dxls[id]
            dxl_addparam_result = self.groupBulkWrite.addParam(id, dxl.ADDR_D, dxl.LEN_PID, [D])
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % id)
                quit()
        # Bulkwrite D values
        dxl_comm_result = self.groupBulkWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear bulkwrite parameter storage
        self.groupBulkWrite.clearParam()

    def update_speed(self, speed: int = 500):
        """ Updates the max speed for all Dynamixels 

        Args:
            speed (int): Max speed for all Dynamixels
        
        Returns:
            none
        """    
        
        # Loop through all Dxls
        param_speed = [DXL_LOBYTE(speed), DXL_HIBYTE(speed)]
        for id in self.dxls.keys():
            # Add Dynamixel max speed value to the Bulkwrite parameter storage
            dxl_addparam_result = self.groupBulkWrite.addParam(id, 32, 2, param_speed)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % id)
                print("wtf")
                quit()
        # Bulkwrite speed vaues
        dxl_comm_result = self.groupBulkWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear bulkwrite parameter storage
        self.groupBulkWrite.clearParam()

    
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
            if self.flag:
                goal = dxl.goal_position + dxl.shift

            else:
                goal = dxl.goal_position


            #dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, id, 30, goal)
            ##if dxl_comm_result != COMM_SUCCESS:
            #    print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            #elif dxl_error != 0:
            #    print("%s" % self.packetHandler.getRxPacketError(dxl_error))


            param_goal_position = [DXL_LOBYTE(goal), DXL_HIBYTE(goal)]
            ## Add Dynamixel goal position value to the Bulkwrite parameter storage
            dxl_addparam_result = self.groupBulkWrite.addParam(id, dxl.ADDR_GOAL_POSITION, 2, param_goal_position) #"""dxl.LEN_GOAL_POSITION"""
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % id)
                quit()
        #self.flag = False
        
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
        if new_goal < self.dxls[id].min_bound:
            new_goal = self.dxls[id].min_bound
        elif new_goal > self.dxls[id].max_bound:
            new_goal = self.dxls[id].max_bound
        self.dxls[id].goal_position = new_goal

        
    def end_program(self):
        """ Turns off Dynamixel torque and closes the port. Run this upon exit/program end.

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
        print(f"here: {file_path}")
        with open(file_path, 'rb') as f:
            self.data = pkl.load(f)


        return len(self.data)

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

    def convert_pos_to_rad(self, pos: int) -> float:
        pos = float(pos)
        deg = np.multiply(pos, (300.0/1023.0))

        rad = np.multiply(deg, (pi/180.0))

        return rad


    def map_pickle(self, i: int):
        """ Convert from relative to absolute positions based on the calibration. Updates global goal position variable.

        Args:
            i (int): Index of the list to convert
        Returns:
            none
        """

        # Set the positions in terms of actual calibrated motor positions
        for id in self.dxls.keys():
            #print(self.data[i])
            self.dxls[id].goal_position = self.dxls[id].center_pos + self.convert_rad_to_pos(self.data[i]["joint_" + str(id+1)])
            #self.dxls[id].joint_angles_pickle[i]


    def replay_pickle_data(self, file_location="Open_Loop_Data", file_name="angles_N.pkl", delay_between_steps: float = .01):
        
        try:    
            # Get our pickle data
            pickle_length = self.load_pickle(file_location, file_name)


            #self.map_pickle(0)
            #self.send_goal()
            #input("Press Enter to continue to next step.")
            self.flag = True
            self.skipp = False
            for i in range(pickle_length):
                if self.skipp == True:
                    self.skipp = False
                    continue
                #if self.event.is_set():
                #    break
                self.map_pickle(i)
                self.send_goal()
                #sleep(delay_between_steps)
                #self.bulk_read_pos()
                self.skipp = True
                #self.bulk_read_pos()

        except KeyboardInterrupt:
            self.end_program()
        self.end_program()

    

    def go_to_initial_position(self, file_location="actual_trajectories_2v2", file_name="N_2v2_1.1_1.1_1.1_1.1.pkl"):
        #try: 
        self.go_to_start()
        sleep(2)
        self.flag = True
        pickle_length = self.load_pickle(file_location, file_name)
        self.map_pickle(0)
        self.send_goal()
        #except:
            #print("ahhh")
            #self.end_program()

    def go_to_start(self):
        #try: 
        for id in self.dxls.keys():
            if id == 0:
                self.update_goal(id, self.dxls[id].center_pos)
            elif id == 2:
                self.update_goal(id, self.dxls[id].center_pos)
            else:
                self.update_goal(id, self.dxls[id].center_pos)
        
        self.send_goal()

        #except:
            #print("ahhh")
            #self.end_program()

    def test_write(self):
        # Start by writing the speed value
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, 0, 24, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#0 has been successfully torqued")

        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, 0, 32, 200)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#0 has been successfully speed")

        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, 0, 30, 550)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#0 has been successfully sent")




if __name__ == "__main__":
    Dynamixel_control = Dynamixel()
    Dynamixel_control.add_dynamixel(ID_number=0, calibration=[0,429,1023], shift = 10) # Negative on left side was -25
    Dynamixel_control.add_dynamixel(ID_number=1, calibration=[0,575,1023], shift = 25)
    Dynamixel_control.add_dynamixel(ID_number=2, calibration=[0,481,1023], shift = 20) # Positive on right side was 25
    Dynamixel_control.add_dynamixel(ID_number=3, calibration=[0,604,1023], shift = -10)
    Dynamixel_control.add_dynamixel(ID_number=4, calibration=[0,309,1023], shift = 0) # Positive on right side was 25
    Dynamixel_control.add_dynamixel(ID_number=5, calibration=[0,497,1023], shift = -20)
    #4565, 545, 450, 553

    Dynamixel_control.setup_all()
    Dynamixel_control.update_PID(85,40,45)
    #Dynamixel_control.update_speed(400)
    #Dynamixel_control.test_write()
    #input("press enter to continue")
    print("PID done")
    Dynamixel_control.update_speed(100)
    #Dynamixel_control.go_to_start()
    sleep(1)
    print("Speed done")
    Dynamixel_control.go_to_initial_position(file_location = "Open_Loop_Data/3v3_50.25.25_25.45.30_1.1_63",file_name="SW_3v3_50.25.25_25.45.30_1.1_63.pkl")
    Dynamixel_control.update_speed(1000)
    print("Speed done")
    input("Press enter")
    Dynamixel_control.replay_pickle_data(file_location = "Open_Loop_Data/3v3_50.25.25_25.45.30_1.1_63",file_name="N_3v3_50.25.25_25.45.30_1.1_63.pkl", delay_between_steps = 0)

    