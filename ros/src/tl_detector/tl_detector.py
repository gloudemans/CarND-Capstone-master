#!/usr/bin/env python
import rospy
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

class TLDetector(object):
    """
    This file implement the traffic light detection ROS node. The node subscribes
    to the following topics:

    * /current_pose - current vehicle twist state
    * /base_waypoints - desired vehicle route
    * /vehicle/traffic_lights - traffic light positions along route
    * /image_color - camera images

    When drive by wire is enabled, the node publishes the following topics:

    * /vehicle/traffic_waypoint - index of upcoming red traffic lights in range 
    
    When the node receives current pose, base waypoints, or traffic light messages 
    it saves their contents for reference. When the node receives an image message, 
    it classifies the image as either containing a red traffic light or not. If the 
    image contains a red traffic light, the node determines the index of the traffic
    light and publishes that index on the traffic waypoint topic. If the image does
    not contain a red light the node published the sentinel value -1. The traffic
    light classifier is implemented as a separate component.
    """

    def __init__(self):
        """
        Construct new object.

        Returns: TLDetector object.

        """

        # Current vehicle pose
        self.pose = None

        # Most recent camera image
        self.camera_image = None

        # List of traffic lights
        self.lights = None

        # Map waypoint list 
        self.waypoints = None

        # Map waypoint list xy only 
        self.waypoints_2d = None

        # Map waypoint list xy only as KDTree
        self.waypoint_tree = None

        # ROS object supporting bidirectionl coversion between ROS image and CV2 images
        self.bridge = CvBridge()

        # ROS object supporting coordinte space transforms
        self.listener = tf.TransformListener()

        # My object for classifying traffic light images as red, green, yellow
        self.light_classifier = TLClassifier()

        # Current state of traffic light
        self.state = TrafficLight.UNKNOWN

        # Prior state of traffic light
        self.last_state = TrafficLight.UNKNOWN

        # Waypoint index of upcoming red light
        self.last_wp = -1

        # Number of consistent detections of light state (for deglitching)
        self.state_count = 0

        '''
        /vehicle/traffic_lights provides a list of all traffic lights. It provides
        their locations in 3D map space and in the simulator it also provides ground
        truth light color state.      
        '''
        # Get traffic light configuration string
        config_string = rospy.get_param("/traffic_light_config")

        # Construct python object from string
        self.config = yaml.load(config_string)

        # Set a node name - something relevant
        rospy.init_node('tl_detector')

        # Add subscriptions and handlers for relevant messages
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_light_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)

        # Create publisher for red lights
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # Keep node alive until ros shutdown
        rospy.spin()

    def current_pose_cb(self, msg):
        """
        Update the current pose.

        Args:
            msg: message containing the current pose.
        """

        # Save the current vehicle pose
        self.pose = msg

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

    def traffic_light_cb(self, msg):

        """
        Save the traffic light array.

        This callback gets executed at a 50 Hz rate set in light publisher.

        This callback receives a list containing the positions of all traffic lights.
        There is also a light state field that contains the actual state of the lights
        when operating in the simulator.

        Args:
            msg: set of traffic lights

        """

        # Save the traffic light array
        self.lights = msg.lights

    def image_cb(self, msg):

        """
        Identifies red lights in the incoming camera image and publishes the index
        of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        # Save the camera image
        self.camera_image = msg

        # I sufficient information is available...
        if not None in (self.camera_image, self.waypoint_tree, self.lights):

            # Find index and color state of next light
            light_wp, state = self.process_traffic_lights()

            # If the light is green...
            if state == TrafficLight.GREEN:

                # Publish sentinel indicatig no red light
                self.upcoming_red_light_pub.publish(Int32(-1))

            else:

                # Publish the traffic light index
                self.upcoming_red_light_pub.publish(Int32(light_wp))

    def process_traffic_lights(self):

        """
        Finds closest visible traffic light, if one exists, and determines its
        location and color.

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # Clear closest light
        closest_light = None

        # Clear closest line index
        line_wp_idx = None
        
        # Load stop line positions from config
        stop_line_positions = self.config['stop_line_positions']

        # If we've received valid pose information...
        if self.pose:

            # Find index of waypoint closest to current car position
            car_wp_idx = self.waypoint_tree.query([self.pose.pose.position.x, self.pose.pose.position.y],1)[-1]

            # Largest possible offset from car to light
            diff = len(self.waypoints.waypoints)

            # For each light in the list of traffic lights...
            for i, light in enumerate(self.lights):

                # Get stop line position i
                line = stop_line_positions[i]

                # Find index of waypoint closest to the light
                wp_idx = self.waypoint_tree.query(line,1)[-1]

                # How many waypoints ahead to light? 
                wps = wp_idx - car_wp_idx

                # If light is ahead of or abreast of car
                # and we haven't encuntered a closer one...
                if wps>=0 and wps<diff:

                    # Keep track of least positive waypoint count
                    diff = wps

                    # Save this light
                    closest_light = light

                    # Save the waypoint index for this light
                    line_wp_idx = wp_idx

        # If there is a traffic light in front of us...
        if closest_light:

            # Get its state
            state = self.get_light_state(closest_light)

            # Return its line index and state
            return line_wp_idx, state

        # Otherwise return sentinel indicating no light
        return -1, TrafficLight.UNKNOWN

    def get_light_state(self, light):
        """
        Determines the current color of the traffic light.

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # If there is no image to process...
        if self.camera_image == None:

            # Don't know the color
            return TrafficLight.UNKNOWN

        else:

            # Convert image format
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # Classify the image
            return self.light_classifier.get_classification(cv_image)

if __name__ == '__main__':

    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
