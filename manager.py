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

class ik_manager:

    def __init__(self, name):
        self.name = name



if __name__ == "__main__":
    # Add here
    print("to add to")
