import math
import numpy as np
from matplotlib import pyplot as plt


# This class serves two functions - it identifies the closest points between the lines, and then uses the geometry to find the actual contact point. 
# It's not a class yet but will be soon lol

# For our standard flat distal with a semicircle at the tip, we can find the conact point by comparing the delta x and delta y from the base of the finger line to the contact point on the object

def find_closest_point(finger_line, object_line):
    """ Finds and returns the closest points between two line segments, and the distance between those points. 
    
    Args:
        finger_line (list) is array of two points similar to [[0.0, 0.0], [1.0, 0.0]] representing the finger line segment
        object_line (list) is array of two points similar to [[0.0, 0.0], [1.0, 0.0]] representing a line segment of the object

    Returns: 

    """

    # Initilize variables for recording the best points
    best_val = 100000.000
    best_intersect_point = [0.0,0.0]
    best_object_point = [0.0, 0.0]
    best_finger_point = [0.0, 0.0]

    pt = [0.0,0.0]
    dst = 0.0
    
    # Check both endpoints of the finger line segment
    for finger_point in finger_line:
        pt, dst = point_segment_distance(finger_point, object_line[0], object_line[1])
        if dst < best_val:
            best_val = dst
            best_object_point = pt
            best_finger_point = finger_point

    # Check both endpoints of the object line segment
    for object_point in object_line:
        pt, dst = point_segment_distance(object_point, finger_line[0], finger_line[1])
        if dst < best_val:
            best_val = dst
            best_finger_point = pt
            best_object_point = object_point

    # Return the best points and distance between them
    return best_object_point, best_finger_point, best_val


def point_segment_distance(p = [0.0, 0.0], line_point_1 = [0.0, 0.0], line_point_2 = [0.0, 0.0]):
    """ Finds the shortest distance between a point and a line. For reference: http://paulbourke.net/geometry/pointlineplane/

    Args:
        p (list): The point we are checking
            (defualt is [0.0, 0.0])
        line_point_1 (list): One endpoint of the line
            (default is [0.0, 0.0])
        line_point_2 (list): The other endpoint of the line
            (default is [0.0, 0.0])
    
    Returns:
        closestPoint (list): The closest point on the line to the provided point
        distance (floast): The distance between the point and closest point
    """

    xdelt = line_point_2[0] - line_point_1[0]
    ydelt = line_point_2[1] - line_point_1[1]

    u = np.round(((p[0]-line_point_1[0])*xdelt+((p[1]-line_point_1[1])*ydelt))/(math.dist(line_point_1, line_point_2)**2),4)

    ix = np.round(line_point_1[0] + (u * xdelt),4)
    iy = np.round(line_point_1[1] + (u * ydelt),4)
    closestPoint = [ix, iy]
    if u < 0.0:
        closestPoint = line_point_1
    elif u >= 1.0:
        closestPoint = line_point_2

    distance = math.dist(p, closestPoint)
    #print(f"line 1: {line_point_1}; line 2: {line_point_2}; point: {p}")
    #print(f"u: {u}; dist: {distance}")
    #print(f"closest: {closestPoint}")

    return closestPoint, distance

def calculate_closest_points(object = [[[0.0,0.0],[2.0, 0.0]], [[2.0,0.0],[2.0, 2.0]], [[2.0,2.0],[0.0, 2.0]], [[0.0,2.0],[0.0, 0.0]]], finger = [[1.2,2.1],[2.1, 2.2]], visualize = False):
    """ Returns the closest points between an object and line

    Args:
        object (list): A list of the 4 sides of the object, defined by their end points
            (default is [[[0.0,0.0],[2.0, 0.0]], [[2.0,0.0],[2.0, 2.0]], [[2.0,2.0],[0.0, 2.0]], [[0.0,2.0],[0.0, 0.0]]])
        finger (list): A list of the start and end point of the finger line
            (default is [[1.2,2.1],[2.1, 2.2]])
        visualize (boolean): Whether or not to plot the object/finger line and closest points
            (default is False)

    Returns:
        best_end_point (list): The point on the end of a line that is closest to the other line
        best_intersect_point (list): The point on the other line (could be an end point or may not be) that is closest to the other end point
    """
    # TODO: Do we actually need to check all 4 sides of the object? Can't we just do 2 of them?

    best_val = None
    best_object_point = [0.0,0.0]
    best_finger_point = [0.0, 0.0]

    for object_line in object:
        obj_point, fing_point, dist = find_closest_point(finger, object_line)

        if not best_val:
            best_val = dist
            best_object_point = obj_point
            best_finger_point = fing_point
        if dist < best_val:
            best_val = dist
            best_object_point = obj_point
            best_finger_point = fing_point
        
    if visualize:
        # Plot the 4 sides of the object
        for object_line in object_side:
            plt.plot([object_line[0][0], object_line[1][0]], [object_line[0][1], object_line[1][1]], "g")

        # Plot the finger line
        plt.plot([finger_line[0][0], finger_line[1][0]], [finger_line[0][1], finger_line[1][1]], "g")

        # Plot the closest point on the object in red
        plt.plot(best_object_point[0], best_object_point[1],'ro')
        # Plot the closest point on the finger line
        plt.plot(best_finger_point[0],best_finger_point[1],'bo')
        plt.show()


    return best_object_point, best_finger_point

def contact_point_calculation(object_size = 39, object_pose = [0.0,0.0,0.0], finger_points = [[0.0,0.0],[1.0,0.0]], distal_length = 72):
    """ Takes in all neccessary parameters and returns the contact point in the distal frame
    
    
    """


# To check all side of the object
object_side = [[[0.0,0.0],[2.0, 0.0]], [[2.0,0.0],[2.0, 2.0]], [[2.0,2.0],[0.0, 2.0]], [[0.0,2.0],[0.0, 0.0]]]

# Define first point of the finger line as closer to the joint and second as the tip of the distal (actually center of the semicircle)
#finger_line = [[1.2,2.1],[2.1, 2.2]]
#finger_line = [[2.1,1],[2.1, 2.2]]
finger_line = [[-1,3.0],[-.5,1.0]]

best_object_point, best_finger_point = calculate_closest_points(object_side, finger_line, True)
