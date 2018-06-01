#!/usr/bin/env python

import rospy

from scipy.spatial import KDTree
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
import numpy as np

import math

'''
Lane:
  Header header
  Waypoint[] waypoints

Waypoint:
  geometry_msgs/PoseStamped pose
  geometry_msgs/TwistStamped twist
'''

# Set loop frequency in Hz
LOOP_FREQUENCY = 50

# Set number of waypoints to publish
LOOKAHEAD_WPS = 50

# Maximum rate of deceleration in m/S^2
MAX_DECEL = 0.5 

class WaypointUpdater(object):
    """
    Implements the waypoint updater ROS node.
    """

    def __init__(self):
        """
        Construct a new waypoint updater ROS node. Subscribe to relevant
        topics and start periodic publishing loop.

        Returns: WaypointUpdater
        """

        # Set a node name - something relevant
        rospy.init_node('waypoint_updater')

        # Most recent pose
        self.pose = None

        # Map waypoint list 
        self.waypoints = None

        # Map waypoint list xy only 
        self.waypoints_2d = None

        # Map waypoint list xy only as KDTree
        self.waypoint_tree = None

        # Index at which to stop the vehicle
        # Negative one is a sentinel meaning no stop is required
        self.stopline_waypoint_idx = -1

        # Add subscriptions and handlers for relevant messages
        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_waypoint_cb)

        # Create publisher for final waypoints
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        # Start loop
        self.loop()

    def loop(self):
        """
        Periodically publish waypoints based on the current vehicle position and desired route.
        """

        # Set loop frequency
        rate = rospy.Rate(LOOP_FREQUENCY)

        # While the system is active...
        while not rospy.is_shutdown():

            # If current pose and waypoints are known...
            if not None in (self.pose, self.waypoints, self.waypoints_2d, self.waypoint_tree):

                # Generate and publish the list of waypoints to follow
                self.publish_waypoints()

            # Wait for next rate tick
            rate.sleep()

    def base_waypoints_cb(self, waypoints):
        """
        Save the base waypoint list, and derive a KDTree object that
        can be used to rapidly find the closest waypoint.

        Args:
            waypoints: waypoints specifying the entire desired route
        """

        # Save the waypoint list
        self.waypoints = waypoints

        # If waypoints_2d hasn't been initialized...
        if not self.waypoints_2d:

            # Extract xy coordinates from the waypoint list
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]

            # Construct a KDTree from the xy coordinate list to allow fast lookup 
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def current_pose_cb(self, msg):
        """
        Update the current pose.

        Args:
            msg: message containing the current pose.
        """

        # Save the current vehicle pose
        self.pose = msg

    def traffic_waypoint_cb(self, msg):
        """
        Update the upcoming red light index..

        Args:
            msg: message containing the index of the closest upcoming
                 red light or -1 if no upcoming red light.
        """

        # Save waypoint index for detected traffic light
        self.stopline_waypoint_idx = msg.data

    def publish_waypoints(self):
        """
        Publish a list containing a fixed number (LOOKAHEAD_WPS) of upcoming 
        waypoints.
        """

        # Make a lane message
        lane = Lane()

        # Get closest waypoint index
        closest_idx = self.get_closest_waypoint_idx()

        # Get farthest waypoint index
        farthest_idx = closest_idx + LOOKAHEAD_WPS

        # Slice to get the upcoming waypoints
        upcoming_waypoints = self.waypoints.waypoints[closest_idx:farthest_idx]

        # If no stopline detected or stopline is beyond farthest index...
        if (self.stopline_waypoint_idx == -1) or (self.stopline_waypoint_idx >= farthest_idx):

            # Follow the upcoming waypoints
            lane.waypoints = upcoming_waypoints

        else:

            # Create a list to hold modified upcoming waypoints
            temp = []

            # Find the relative stopline index within the upcoming waypoints
            # Back off by two waypoints so that front of car stays behind
            # stopline.
            stop_idx = max(self.stopline_waypoint_idx-closest_idx-2, 0)

            # Get the deceleration velocities at each upcoming waypoint
            velocities = self.deceleration_velocities(upcoming_waypoints, stop_idx)

            # For each upcoming waypoint...
            for i, wp in enumerate(upcoming_waypoints[:-1]):

                # Create a new waypoint
                p = Waypoint()

                # Dupicate the pose of the existing waypoint
                p.pose = wp.pose

                # Limit current velocities to decelration velocities
                p.twist.twist.linear.x = min(velocities[i], p.twist.twist.linear.x)

                # Add the modified waypoint to the list
                temp.append(p)

            # Follow the modified upcoming waypoints
            lane.waypoints = temp

        # Publish the lane message
        self.final_waypoints_pub.publish(lane)

    def get_closest_waypoint_idx(self):
        """
        Find the index of the waypoint closest to the current vehicle position. 
        """

        # TODO:
        # The churchlot waypoints are roughly circular but have self-
        # intersecting endpoints, so I'm not sure how this code will 
        # yield good results. Might need some additional filtering
        # logic to force a choice consistent with the vehicle pose yaw
        # in order to avoid jumping onto the wrong path.

        # Vehicle position short reference
        pos = self.pose.pose.position

        # Find the closest waypoint index
        # If closest index is zero bump to 1 since we don't want slice for 
        # prev_coord to look at the final map waypoint.
        closest_idx = max(self.waypoint_tree.query([pos.x, pos.y], 1)[1], 1)

        # Get closest point
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Convert coordinates into 2D  numpy vectors
        closest_vec = np.array(closest_coord)
        prev_vec = np.array(prev_coord)
        pos_vec = np.array([pos.x, pos.y])

        # Find vec(close-prev) dot vec(pos-close) 
        val = np.dot(closest_vec - prev_vec, pos_vec - closest_vec)

        # If pos is ahead of closest...
        if val > 0:   

            # Advance index so that closest is ahead of pos
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        # Return closest index
        return closest_idx 

    def deceleration_velocities(self, waypoints, stop_idx):
        """
        Given a list of waypoints and the index of a waypoint at which to
        stop the vehicle, compute the maximum velocities at each waypoint
        from the stoplight at which the deceleration limit can be honored
        while stopping at the light.

        Args: 
            waypoints: list of waypoints through which to pass
            stop_idx: index of the waypoint at which to stop

        Returns: velocities at each waypoint
        """

        # TODO:
        # It would be cool to calculate the jerk minimizing trajectory as the velocity profile
        # here instead of using uniform accelaration. I think this might be straightforward
        # and it would give the test passengers a nice smooth ride. Consider doing this if there 
        # is time.

        # Get waypoint xyz coordinates as np array
        xyz = np.asarray([ [wp.pose.pose.position.x, wp.pose.pose.position.y, wp.pose.pose.position.z] for wp in waypoints])

        # Compute the cumulative distance between points
        cumulative_distances = np.cumsum(np.sqrt(np.sum(np.square(xyz[1:,:] - xyz[:-1,:]), axis = -1)))

        # Compute relative distance to stopping point
        stop_distances = np.maximum(0, cumulative_distances[stop_idx] - cumulative_distances)

        # Compute velocity needed to cause uniform deceleration
        velocities = np.sqrt(2*MAX_DECEL*stop_distances)

        # Return the desired deceleration velocity at each point
        return velocities

if __name__ == '__main__':

    try:

        WaypointUpdater()

    except rospy.ROSInterruptException:

        rospy.logerr('Could not start waypoint updater node.')
