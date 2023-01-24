#!/usr/bin/env python3

# The usual imports
import numpy as np
import matplotlib.pyplot as plt

# If this doesn't work, right click on top level folder and pick "mark folder" as source
import arm_forward_kinematics as afk
import arm_ik_gradient as ik_gradient


# BEGIN SOLUTION NO PROMPT
# Just in case I accidentally include this file
def test_reach_target(fc):
    """Do a slightly different arm geometry and test reaches target"""
    # Create the arm geometry
    base_size_param = (1.0, 0.5)
    link_sizes_param = [(0.5, 0.25), (0.3, 0.1), (0.2, 0.05), (0.1, 0.025)]
    palm_width_param = 0.1
    finger_size_param = (0.075, 0.025)

    arm_geometry = afk.create_arm_geometry(base_size_param, link_sizes_param, palm_width_param, finger_size_param)
    angles_check = [0.0, np.pi/16, np.pi/32, -np.pi/64, [-np.pi/8.0, np.pi/8.0, np.pi/16.0]]
    afk.set_angles_of_arm_geometry(arm_geometry, angles_check)

    for x in np.linspace(-0.75, 0.75, 3):
        for y in np.linspace(0.75, 1.3, 3):
            target = (x, y)
            afk.set_angles_of_arm_geometry(arm_geometry, angles_check)
            b_succ, angles_new, count = fc(arm_geometry, angles_check, target, b_one_step=False)
            afk.set_angles_of_arm_geometry(arm_geometry, angles_new)
            dist = ik_gradient.distance_to_goal(arm_geometry, target)
            if dist > 0.01:
                return False
    return True
# END SOLUTION

