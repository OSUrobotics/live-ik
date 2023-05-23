
from sys import exit as ex
import numpy as np

class Dxl:

    def __init__(self, dxl_dict): 
        calibration = dxl_dict["calibration"]
        self.type = dxl_dict["type"] 

        if dxl_dict["type"] == "XL-320":
            self.dxl_params = {"ADDR_P_position": 29,
                               "ADDR_I_position": 28,
                               "ADDR_D_position": 27,
                               "LEN_PID_position": 1,

                               "ADDR_torque_enable": 24,
                               "LEN_torque_enable": 1,

                               "ADDR_goal_position": 30,
                               "LEN_goal_position": 4,

                               "ADDR_present_position": 37,
                               "LEN_present_position": 2
            }
        elif dxl_dict["type"] == "XL-330":
            self.dxl_params = {"ADDR_P_position": 84,
                                "ADDR_I_position": 80,
                                "ADDR_D_position": 82,
                                "LEN_PID_position": 2,

                                "ADDR_torque_enable": 64,
                                "LEN_torque_enable": 1,

                                "ADDR_velocity_cap": 112,
                                "LEN_velocity_cap": 4,

                                "ADDR_goal_position": 116,
                                "LEN_goal_position": 4,

                                "ADDR_present_position": 132,
                                "LEN_present_position": 4
            }
            print("This worked")
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


