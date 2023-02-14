import numpy as np

def get_square_pose(corners):
    # Find the midpoint between the first and third corner
    first_corn = corners[0]
    third_corn = corners[2]

    delta_x = third_corn[0] - first_corn[0]
    delta_y = third_corn[1] - first_corn[1]

    # This is the midpoint of the line, which is the center of the object
    midpoint = [first_corn[0]+delta_x/2.0, first_corn[1]+delta_y/2.0]
    print(midpoint)

    # Now we find the rotation, define CCW as + (with the unit circle)
    angle = rotate_pose(first_corn, midpoint)

    return [midpoint[0], midpoint[1], angle]

def rotate_pose(first_corner, center):
    # Define unit vector we care about (we can use this reference to calculate contact deltas in the correct frame)
    def_vec = [0, 1] # Vertical along y-axis

    # Get line deltas
    d_x = first_corner[0]-center[0]
    d_y = first_corner[1]-center[1]

    # Get vector
    in_vector = [d_x, d_y]

    # Normalize vector
    unit_vec = in_vector / np.linalg.norm(in_vector)

    # Find the angle between vectors
    dot_product = np.dot(def_vec, unit_vec)
    angle = np.arccos(dot_product)

    if first_corner[0] < center[0]:
        # if in left quadrants, positive rotation
        angle = angle + np.pi/4
    else:
        # Negative angle 
        angle = -angle - np.pi/4
        
    return angle


corn = [[1, 1], [1, 0], [0, 0], [0, 1]]
get_square_pose(corn)

