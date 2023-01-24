from dynamixel_sdk import *                    # Uses Dynamixel SDK library   
import dynamixel
from time import sleep

# Operates on the assumption that index 0 = dxl 0 and so on...

# TODO: tune PID https://www.youtube.com/watch?v=msWlMyx8Nrw&ab_channel=ROBOTISOpenSourceTeam

class Dynamixel:

    def __init__(self): 
        self.DEVICENAME = '/dev/ttyUSB0'
        self.PROTOCOL_VERSION = 2.0
        self.BAUDRATE = 57600

        self.portHandler = PortHandler(self.DEVICENAME)
        

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

        # Create a list of the Dynamixels
        self.dxls = []
            
    def create_dynamixel_dict(self, type = "XL-320", ID_number = 0, calibration = [0, 511, 1023]) -> dict:
        """Takes in parameters and creates a dictionary for the dynamixel.

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
                          "calibration": calibration}
        return dynamixel_dict

    def add_dynamixel(self, dict):
        """Adds a tuple to the 

        Args:
            i (int): Position in list of Dynamixels
        Returns:
            none
        """
        self.dxls.append(dynamixel.Dxl(dict)) 

    def enable_torque(self, i: int):
        """Enables torque for one Dynamixel.

        Args:
            i (int): Position in list of Dynamixels
        Returns:
            none
        """

        # Enable Dynamixel Torque
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dxls[i].dxl_ID, self.dxls[i].ADDR_TORQUE_ENABLE, self.dxls[i].TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully connected" % self.dxls[i].dxl_ID)

    def setup_position(self, i):
        # Add parameter storage for Dynamixel#0 present position
        dxl_addparam_result = self.groupBulkRead.addParam(self.dxls[i].dxl_ID, self.dxls[i].ADDR_PRESENT_POSITION, self.dxls[i].LEN_PRESENT_POSITION)
        if dxl_addparam_result != True:
            print("[ID:%03d] groupBulkRead addparam failed" % self.DXL0_ID)
            quit()

    def setup_all(self):
        for i in range(len(self.dxls)):
            self.enable_torque(i)
        
        for i in range(len(self.dxls)):
            self.setup_position(i)

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
        for i, dxl in enumerate(self.dxls):
            dxl.read_position = self.groupBulkRead.getData(dxl.dxl_ID, dxl.ADDR_PRESENT_POSITION, 2)#self.LEN_PRESENT_POSITION)
            print(dxl.read_position)

    def send_goal(self):
        """ Writes goal positions to all Dynamixels. Based on goal position stored in each Dxl object.

        Args:
            none
        
        Returns:
            none
        
        """

        for i, dxl in enumerate(self.dxls):
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(dxl.goal_position)), DXL_HIBYTE(DXL_LOWORD(dxl.goal_position)), DXL_LOBYTE(DXL_HIWORD(dxl.goal_position)), DXL_HIBYTE(DXL_HIWORD(dxl.goal_position))]
            # Add Dynamixel goal position value to the Bulkwrite parameter storage
            dxl_addparam_result = self.groupBulkWrite.addParam(dxl.dxl_ID, dxl.ADDR_GOAL_POSITION, dxl.LEN_GOAL_POSITION, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkWrite addparam failed" % dxl.dxl_ID)
                quit()
        
        # Bulkwrite goal position and LED value
        dxl_comm_result = self.groupBulkWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear bulkwrite parameter storage
        self.groupBulkWrite.clearParam()
    
    def update_goal(self, id_number: int, new_goal: int):
        """ Updates the goal position stored in the object for 1 dynamixel

        Args:
            id_number (int): ID number of Dynamixel to update
            new_goal (int): New goal position from 0 to 1023
        Returns:
            none
        
        """
        self.dxls[id_number].goal_position = new_goal

        
    def end_program(self):
        """ Turns off Dynamixel torque and closes the port. Run this upone exit/program end.

        Args:
            none
        Returns:
            none
        
        """

        # Clear bulkread parameter storage
        self.groupBulkRead.clearParam()

        for i in range(len(self.dxls)):
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, self.dxls[i].dxl_ID, self.dxls[i].ADDR_TORQUE_ENABLE, self.dxls[i].TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d has been successfully connected" % self.dxls[i].dxl_ID)

        self.portHandler.closePort()  



if __name__ == "__main__":
    Dynamixel_control = Dynamixel()
    for i in range(4):
        Dynamixel_control.add_dynamixel(Dynamixel_control.create_dynamixel_dict(ID_number=i))

    Dynamixel_control.setup_all()

    
    Dynamixel_control.bulk_read_pos()
    Dynamixel_control.send_goal()
    Dynamixel_control.bulk_read_pos()
    Dynamixel_control.end_program()