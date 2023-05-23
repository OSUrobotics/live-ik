
from sys import exit as ex
import numpy as np

class Dxl:

    def __init__(self, dxl_dict): 
        calibration = dxl_dict["calibration"]
        self.type = dxl_dict["type"] 

        if dxl_dict["type"] == "XL-320":
            # Table values that corespond to the Dynamixel XL-320
            # PID Values
            self.ADDR_P_position             = 29  # 0 to 254
            self.ADDR_D_position             = 27
            self.ADDR_I_position             = 28
            self.LEN_PID                     = 1
            # Max Torque
            self.max_torque                  = 1023     # 0 to 1023
            self.ADDR_max_torque             = 15
            self.LEN_max_torque              = 2


            self.ADDR_TORQUE_ENABLE          = 24
            self.ADDR_LED_RED                = 65
            self.LEN_LED_RED                 = 1         # Data Byte Length
            self.ADDR_GOAL_POSITION          = 30
            self.LEN_GOAL_POSITION           = 4         # Data Byte Length
            self.ADDR_PRESENT_POSITION       = 37
            self.LEN_PRESENT_POSITION        = 2        # Data Byte Length
            self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
            self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
            self.BAUDRATE                    = 57600

            self.CURRENT_TORQUE_INDEX        = 41       # No units - from range 0 to 1023. Technically I have it capped at 950 now tho
            self.LEN_CURRENT_TORQUE_INDEX    = 1

        elif dxl_dict["type"] == "XL-330":
                       # Table values that corespond to the Dynamixel XL-320
            # PID Values
            self.ADDR_P_position             = 84  # 0 to 16,383
            self.ADDR_D_position             = 82
            self.ADDR_I_position             = 80
            self.LEN_PID                     = 2
            # Max Torque
            #self.max_torque                  = 1023     # 0 to 1023
            #self.ADDR_max_torque             = 15
            #self.LEN_max_torque              = 2

            self.ADDR_velocity_profile       = 112
            self.LEN_velocity_profile        = 4


            self.ADDR_TORQUE_ENABLE          = 64
            self.ADDR_LED_RED                = 65
            self.LEN_LED_RED                 = 1         # Data Byte Length
            self.ADDR_GOAL_POSITION          = 116
            self.LEN_GOAL_POSITION           = 4         # Data Byte Length
            self.ADDR_PRESENT_POSITION       = 132
            self.LEN_PRESENT_POSITION        = 4        # Data Byte Length
            self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
            self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
            self.BAUDRATE                    = 57600

            #self.CURRENT_TORQUE_INDEX        = 41       # No units - from range 0 to 1023. Technically I have it capped at 950 now tho
            #self.LEN_CURRENT_TORQUE_INDEX    = 1
        else:
            print(dxl_dict["type"])
            ex("Dynamixel type not implemented")

        
        # Set the dynamixel ID number
        self.dxl_ID = dxl_dict["ID_number"]

        # Set the calibrated center position and boundaries of the dynamixel
        self.min_bound = calibration[0]
        self.center_pos = calibration[1]
        self.max_bound = calibration[2]

        # General settings, generally not touched
        self.TORQUE_ENABLE               = 1                 # Value for enabling the torque
        self.TORQUE_DISABLE              = 0                 # Value for disabling the torque
        self.DXL_MOVING_STATUS_THRESHOLD = 1                 # Dynamixel moving status threshold

        # Set the starting goal position as the calibrated center positions
        self.read_position = calibration[1]
        self.read_position_rad = 0.0
        self.goal_position = calibration[1]
        self.current_torque = 0

        self.read_position_m = None

        self.joint_angles_pickle = np.array([])

        # Add a value shift to account for applying force
        self.shift = dxl_dict["shift"]


