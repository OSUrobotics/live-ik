import math
import numpy as np
from matplotlib import pyplot as plt
import sys

# This class serves two functions - it identifies the closest points between the lines, and then uses the geometry to find the actual contact point. 

# For our standard flat distal with a semicircle at the tip, we can find the conact point by comparing the delta x and delta y from the base of the finger line to the contact point on the object
class ContactPoint:
    def __init__(self, object_size = 39.0, distal_length = 72.0, sleeve_length = 50.0, side = "L") -> None:
        # Set up variables for future use

        # Set up object side length (in mm)
        self.object_size = object_size

        # Determine the correct distance between joint and red line
        # IF MORE SLEEVES CREATED, ADD THEM HERE
        if np.isclose(sleeve_length, 30):
            spacing = 10
        elif np.isclose(sleeve_length, 50):
            spacing = 15
        else: 
            sys.exit("Error - incorrect sleeve length")

        self.dist_joint_to_red = distal_length - sleeve_length + spacing
        self.first = True

        self.side = "L"

        
    def find_closest_point(self, finger_line, object_line):
        """ Finds and returns the closest points between two line segments, and the distance between those points. 
        
        Args:
            finger_line (list): An array of two points similar to [[0.0, 0.0], [1.0, 0.0]] representing the finger line segment
            object_line (list): An array of two points similar to [[0.0, 0.0], [1.0, 0.0]] representing a line segment of the object

        Returns: 
            best_object_point (list): The closest point on the object, represented as [x, y]
            best_finger_point (list): The closest point on the finger, represented as [x, y]
            best_val (float): The distance between the closest point on the object and finger
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
            pt, dst = self.point_segment_distance(finger_point, object_line[0], object_line[1])
            if dst < best_val:
                best_val = dst
                best_object_point = pt
                best_finger_point = finger_point

        # Check both endpoints of the object line segment
        for object_point in object_line:
            pt, dst = self.point_segment_distance(object_point, finger_line[0], finger_line[1])
            if dst < best_val:
                best_val = dst
                best_finger_point = pt
                best_object_point = object_point

        # Return the best points and distance between them
        return best_object_point, best_finger_point, best_val


    def point_segment_distance(self, p = [0.0, 0.0], line_point_1 = [0.0, 0.0], line_point_2 = [0.0, 0.0]):
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

    def calculate_closest_points(self, object = [[[0.0,0.0],[2.0, 0.0]], [[2.0,0.0],[2.0, 2.0]], [[2.0,2.0],[0.0, 2.0]], [[0.0,2.0],[0.0, 0.0]]], finger = [[1.2,2.1],[2.1, 2.2]], visualize = False):
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
            obj_point, fing_point, dist = self.find_closest_point(finger, object_line)

            if not best_val:
                best_val = dist
                best_object_point = obj_point
                best_finger_point = fing_point
            if dist < best_val:
                best_val = dist
                best_object_point = obj_point
                best_finger_point = fing_point
        
        if visualize and not self.first:
            # Plot the 4 sides of the object
            plt.clf()
            for object_line in object:
                plt.plot([object_line[0][0], object_line[1][0]], [object_line[0][1], object_line[1][1]], "g")

            # Plot the finger line
            plt.plot([finger[0][0], finger[1][0]], [finger[0][1], finger[1][1]], "g")

            # Plot the closest point on the object in red
            plt.plot(best_object_point[0], best_object_point[1],'ro')
            # Plot the closest point on the finger line
            plt.plot(best_finger_point[0],best_finger_point[1],'bo')
            plt.show()


        return best_object_point, best_finger_point

    def create_object_segments(self, object_size = 39.0, object_pose = [0.0, 0.0, 0.0]):
        """ Returns a list of line segments of the actual object size/pose

        Args:
            object_size (float): Object side length (assuming a square) in mm
                (default is 39.0)
            object_pose (list): A list of the object pose (x, y, theta) in meters and radians
                (default is [0.0, 0.0, 0.0])

        Returns:
            output (list): The four line segments of the object
        """

        # Create the base points to transform
        base_object_points = np.array([[-.5,.5,.5,-.5],[-.5,-.5,.5,.5],[1.0,1.0,1.0,1.0]])

        # Create a scale matrix to scale to object size
        # Object size input is in mm, converting to meters
        scale_matrix = np.identity(3)
        scale_matrix[0][0] = object_size/1000.0
        scale_matrix[1][1] = object_size/1000.0

        # Create a rotation matrix to rotate to correct orientation
        angle = object_pose[2] # in radians
        rotation_matrix = np.identity(3)
        rotation_matrix[0][0] = np.cos(angle)
        rotation_matrix[0][1] = -np.sin(angle)
        rotation_matrix[1][0] = np.sin(angle)
        rotation_matrix[1][1] = np.cos(angle)

        # Create a translation matrix to move the object to the correct location
        trans_matrix = np.identity(3)
        trans_matrix[0][2] = object_pose[0]
        trans_matrix[1][2] = object_pose[1]

        # Do scale and rotation of points
        object_points = np.matmul(trans_matrix,np.matmul(rotation_matrix, np.matmul(scale_matrix, base_object_points)))

        # Convert into format used elsewhere (line segments)
        output = [#[[object_points[0][0], object_points[1][0]], [object_points[0][1], object_points[1][1]]], 
            [[object_points[0][1], object_points[1][1]], [object_points[0][2], object_points[1][2]]], 
            #[[object_points[0][2], object_points[1][2]], [object_points[0][3], object_points[1][3]]], 
            [[object_points[0][3], object_points[1][3]], [object_points[0][0], object_points[1][0]]]]

        #plt.plot(object_points[0],object_points[1])
        #plt.show()

        return output


    def finger_point_ordering(self, finger_points, finger_joint_angles):
        """ Converts the finger point array to have the first point always be the base of the link

        Args:
            finger_points (list): List of finger points corresponding to red line detected
            finger_joint_angles (list): A list of the joint angles for that finger, all based off of center of previous link

        Returns:
            output_finger_points (list): The finger point array reordered as necessary 
        """

        # Angle of Distal to world is angle of prox + angle of distal from center (for 3 link, additionally link also sums)
        world_ang = np.sum(finger_joint_angles)

        # Create array to store output finger points
        output_finger_points = np.zeros([2, 2])

        # If the angle to world is 0, then straight up relative to the palm
        # If angle to world is > 0 (positive), then angled inwards
        # If angle to world is < 0 (negative), then angled outwards

        # Check if within .4 rad
        if np.abs(world_ang) <= (np.pi/2 +.4) and np.abs(world_ang) >= (np.pi/2 -.4):
            # Then we set it based on x instead of y
            if self.side == "L":
                # X should be more negative for the base
                if finger_points[0][0] <= finger_points[1][0]:
                    # Check if order of points is currently correct (first point is close to joint, second is tip)
                    output_finger_points = finger_points
                else: 
                    # Flip orientation of points
                    output_finger_points[0][:] = finger_points[1][:]
                    output_finger_points[1][:] = finger_points[0][:]
            if self.side == "R":
                # X should be more positive for the base
                if finger_points[0][0] >= finger_points[1][0]:
                    # Check if order of points is currently correct (first point is close to joint, second is tip)
                    output_finger_points = finger_points
                else: 
                    # Flip orientation of points
                    output_finger_points[0][:] = finger_points[1][:]
                    output_finger_points[1][:] = finger_points[0][:]
            return output_finger_points


        
        # Essentially, just get the first point to be associated to the base of the link (joint to previous link)
        if np.abs(world_ang) <= np.pi/2:
            # Standard finger configuration, tip is more positive than base
            if finger_points[0][1] <= finger_points[1][1]:
                # Check if order of points is currently correct (first point is close to joint, second is tip)
                output_finger_points = finger_points
            else: 
                # Flip orientation of points
                output_finger_points[0][:] = finger_points[1][:]
                output_finger_points[1][:] = finger_points[0][:]
            pass
        else:
            # Our base is "more positive" than tip
            if finger_points[0][1] >= finger_points[1][1]:
                # Check if order of points is currently correct (second point is close to joint, first is tip)
                output_finger_points = finger_points
            else: 
                # Flip orientation of points
                output_finger_points[0][:] = finger_points[1][:]
                output_finger_points[1][:] = finger_points[0][:]

        return output_finger_points

    def calculate_contact_delta(self, finger_array, dist_to_base, contact_point):
        """ Finds the deltas from the points of the finger line, distance from joint to finger line, and contact point

        Args:
            finger_array (list): List of finger points corresponding to red line detected
            dist_to_base (float): Distance from the joint on the distal link to the bottom of the red line (in mm)
            contact_point (list): The contact point on the object

        Returns:
            contact_point_delta (list): The [x, y] delta of the contact point in the frame of the distal link (x should be negative, y positive in all cases)
        """

        # Define unit vector we care about (we can use this reference to calculate contact deltas in the correct frame)
        def_vec = [0, 1] # Vertical along y-axis

        # Get finger line deltas
        d_x = finger_array[1][0]-finger_array[0][0]
        d_y = finger_array[1][1]-finger_array[0][1]

        # Get finger vector
        finger_vector = [d_x, d_y]

        # Normalize finger vector
        unit_finger = finger_vector / np.linalg.norm(finger_vector)

        # Find the angle between vectors
        dot_product = np.dot(def_vec, unit_finger)
        angle = np.arccos(dot_product)

        if d_x <= 0.0:
            # In left quadrants, want to rotate negative theta
            angle = -angle
        else:
            # In right quadrants, want to rotate positive theta
            pass

        # Use the angle to create a rotation matrix and rotate the two points
        rot_mat = np.zeros([2,2])
        rot_mat[0][0] = np.cos(angle)
        rot_mat[0][1] = -np.sin(angle)
        rot_mat[1][0] = np.sin(angle)
        rot_mat[1][1] = np.cos(angle)

        # Combine the points in an array of (column points)
        points = np.zeros([2,2])
        points[0][0] = finger_array[0][0]
        points[1][0] = finger_array[0][1]
        points[0][1] = contact_point[0]
        points[1][1] = contact_point[1]

        rotated = np.matmul(rot_mat, points)

        # Factor in actual distance from joint to bottom of red
        rotated[1][0] = rotated[1][0] - (dist_to_base)

        # Contact point in an array of x, y (x must be negative and y positive)
        contact_point_delta = np.zeros(2)
        contact_point_delta[0] = rotated[0][1]-rotated[0][0]
        contact_point_delta[1] = rotated[1][1]-rotated[1][0]

        #print(f"Right finger array: {finger_array}, delta: {contact_point_delta}, contact: {contact_point}")

        return contact_point_delta


    def contact_point_calculation(self, object_pose = [0.0,0.0,0.0], finger_points = [[0.0,0.0],[1.0,0.0]], joint_angles = [0.0, 0.0], side = "L", dist_length=.072, sleeve_length=.0500):
        """ Uses all of the other finctions to find the contact point and return the contact point delta in the distal frame

        Args:
            object_pose (list): The object pose in meters and radians [x, y, theta]
                (default is [0.0, 0.0, 0.0])
            finger_points (list): The two points defining the top/bottom of the red line
                (default is [[0.0, 0.0], [0.0, 0.0]])
            joint_angles (list): All of the joint angles for that link in radians (i.e. [prox_angle, int_angle, dist_angle]). Can accept any length (for 1-infinite number of links)
                (default [0.0, 0.0])

        Returns:
            contact_delta (list): The [x, y] delta of the contact point in the frame of the distal link (x should be negative, y positive in all cases)
        """
        #if side == "L":
        #    print(f"Finger points: {finger_points}")
        if np.isclose(sleeve_length, .030):
            spacing = .010
        elif np.isclose(sleeve_length, .050):
            spacing = .015
        else: 
            sys.exit("Error - incorrect sleeve length")

        self.dist_joint_to_red = dist_length - sleeve_length + spacing
        #print(self.dist_joint_to_red)
        self.side = side
        # Create the object segments
        object = self.create_object_segments(self.object_size, object_pose)

        # Reorder finger points
        finger_points = self.finger_point_ordering(finger_points, joint_angles)

        # Calculate contact point
        contact_point, _ = self.calculate_closest_points(object, finger_points, False)

        
        # Get contact point in distal frame
        contact_delta = self.calculate_contact_delta(finger_points, self.dist_joint_to_red, contact_point)
        self.first = False
        return contact_point, contact_delta

    def project_line(self, line, spacing, dist_length):

        # Points should already be in order maybe??

        # Create vector between the two points
        vector = line[1,:] - line[0,:]
        unit_vector = vector / np.linalg.norm(vector)

        # Distance to top is point + length of spacing
        top_point = line[1,:] + spacing*unit_vector

        # Distance to bottom point is top point - dist length
        bottom_point = top_point - dist_length*unit_vector

        return bottom_point, top_point



    def joint_point_calculation(self, object_pose = [0.0,0.0,0.0], finger_points = [[0.0,0.0],[1.0,0.0]], joint_angles = [0.0, 0.0], side = "L", dist_length=.072, sleeve_length=.0500):
        """ Uses all of the other finctions to find the contact point and return the contact point delta in the distal frame

        Args:
            object_pose (list): The object pose in meters and radians [x, y, theta]
                (default is [0.0, 0.0, 0.0])
            finger_points (list): The two points defining the top/bottom of the red line
                (default is [[0.0, 0.0], [0.0, 0.0]])
            joint_angles (list): All of the joint angles for that link in radians (i.e. [prox_angle, int_angle, dist_angle]). Can accept any length (for 1-infinite number of links)
                (default [0.0, 0.0])

        Returns:
            contact_delta (list): The [x, y] delta of the contact point in the frame of the distal link (x should be negative, y positive in all cases)
        """
        
        if np.isclose(sleeve_length, .030):
            spacing = .010
        elif np.isclose(sleeve_length, .050):
            spacing = .015
        else: 
            sys.exit("Error - incorrect sleeve length")

                # Reorder finger points
        finger_points = self.finger_point_ordering(finger_points, joint_angles)

        # Calculate the bottom and top points
        bottom, top = self.project_line(finger_points, spacing, dist_length)
        
        #if side == "R":
            #print(f"Bottom {bottom}, Top: {top}")
        return bottom, top



if __name__ == "__main__":
    contact = ContactPoint(object_size=39, distal_length=72, sleeve_length=50)

    contact.contact_point_calculation([0.0,0.05,0.3], )
    #print(contact.calculate_contact_delta([[0.0,0.0],[0,1]], 15, [-1,1]))

    #out = contact.finger_point_ordering([[2.0,3.0],[1.0,2.0]], [np.pi/2])
   
    #object = contact.create_object_segments(39,[.2,.1,np.pi/10])

    # To check all side of the object
    #object_side = [[[0.0,0.0],[2.0, 0.0]], [[2.0,0.0],[2.0, 2.0]], [[2.0,2.0],[0.0, 2.0]], [[0.0,2.0],[0.0, 0.0]]]

    # Define first point of the finger line as closer to the joint and second as the tip of the distal (actually center of the semicircle)
    #finger_line = [[1.2,2.1],[2.1, 2.2]]
    #finger_line = [[2.1,1],[2.1, 2.2]]
    #finger_line = [[-1,3.0],[-.5,1.0]]

    #best_object_point, best_finger_point = contact.calculate_closest_points(object, finger_line, True)
