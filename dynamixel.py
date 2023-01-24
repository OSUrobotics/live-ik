
from sys import exit as ex

class Dxl:

    def __init__(self, dxl_dict): 
        calibration = dxl_dict["calibration"]

        if dxl_dict["type"] == "XL-320":
            # Table values that corespond to the Dynamixel XL-320
            self.ADDR_TORQUE_ENABLE          = 24
            self.ADDR_LED_RED                = 65
            self.LEN_LED_RED                 = 1         # Data Byte Length
            self.ADDR_GOAL_POSITION          = 30
            self.LEN_GOAL_POSITION           = 4         # Data Byte Length
            self.ADDR_PRESENT_POSITION       = 37
            self.LEN_PRESENT_POSITION        = 4         # Data Byte Length
            self.DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
            self.DXL_MAXIMUM_POSITION_VALUE  = 4095      # Refer to the Maximum Position Limit of product eManual
            self.BAUDRATE                    = 57600

            # Protocol Version 2.0 for the Dynamixel XL-320
            self.PROTOCOL_VERSION            = 2.0
        else:
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

