#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math
from twist_controller import Controller

# Set loop frequency in Hz
LOOP_FREQUENCY = 50

class DBWNode(object):
    """
    Implements an object that constructs the rospy dbw_node. The dbw_node is responsible
    for converting proposed linear and angular velocities to throttle, brake, and steering 
    commands. This translation is vehicle specific, making use of various physical properties
    of the car. The node subscribes to /current_velocity, /twist_cmd, and 
    /vehicle/dbw_enabled messages to receive current velocity, proposed velocity and acceleration,
    and drive by wire enable status respectively. When drive by wire is enabled, it publishes
    the resulting control messages on /vehicle/steering_cmd, /vehicle/throttle_cmd, and 
    /vehicle/brake_cmd.

    This object acts principally as a rospy interface wrapper for a Controller object that
    performs the business logic.
    """

    def __init__(self):
        """
        Constructs the ROS drive by wire node. Subsribes to relevant topics, 
        and starts a loop to publish throttle brake and steering commands at
        a 50 Hz rate. 

        Returns: new DBWNode
        """

        # Set a node name - something relevant
        rospy.init_node('dbw_node')

        # Get various vehicle parameters
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5.0)
        accel_limit = rospy.get_param('~accel_limit', 1.0)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.0)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.0)
                                        
        # Publisher for throttle (normalized 0 to 1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)

        # Publisher for braking torque (newton meters)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        # Publisher for steering angle (radians)
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)                              

        # Create a parameter pile for the controller constructor
        parameters = {
            'vehicle_mass':vehicle_mass,					
            'fuel_capacity':fuel_capacity,					
            'brake_deadband':brake_deadband,
            'decel_limit':decel_limit,
            'accel_limit':accel_limit,
            'wheel_radius':wheel_radius,
            'wheel_base':wheel_base,
            'steer_ratio':steer_ratio,
            'max_lat_accel':max_lat_accel,
            'max_steer_angle':max_steer_angle									
        }		

        # Create controller object  
        self.controller = Controller(**parameters)

        # Initialize to a benign state
        self.proposed_linear_velocity = None
        self.proposed_angular_velocity = None
        self.current_linear_velocity = None
        self.dbw_enabled = False

        # Add subscriptions and handlers for relevant messages
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        # Start looping
        self.loop()

    def loop(self):
        """
        Start a 50 Hz loop the uses a controller object to compute new throttle brake and 
        steering values based on proposed linear velocity, proposed angular velocity,
        current linear velocity, drive by wire state, and current ROS time. Publish 
        the throttle, brake, and steering values if drive by wire is enabled. 
        """

        # Set loop frequency
        rate = rospy.Rate(LOOP_FREQUENCY) 

        # While the system is active...
        while not rospy.is_shutdown():

            # If we've received sufficient information...
            if not None in (self.proposed_linear_velocity, self.proposed_angular_velocity, self.current_linear_velocity):

                # Run the controller to determine the desired throttle, brake and
                # steering states.
                throttle, brake, steering = self.controller.control(
                    self.proposed_linear_velocity,
                    self.proposed_angular_velocity,
                    self.current_linear_velocity,
                    self.dbw_enabled,
                    rospy.get_time())

                # If the drive by wire system is enabled...
                if self.dbw_enabled:

                    # Publish the actuator state updates 
                    self.publish(throttle, brake, steering)

            # Wait for the next rate tick
            rate.sleep()

    def current_velocity_cb(self, msg):
        """
        Update the current linear velocity.

        Args:
            msg: twist message containing the current linear velocity
        """

        # Extract current velocity from TwistStamped message
        self.current_linear_velocity = msg.twist.linear.x	

    def twist_cb(self, msg):
        """
        Update the proposed linear velocity and angular velocity.

        Args:
            msg: twist message containing the current linear velocity and angular velocity
        """

        # Extract linear velocity from TwistStamped message
        self.proposed_linear_velocity  = msg.twist.linear.x

        # Extract angular velocity from TwistStamped message
        self.proposed_angular_velocity = msg.twist.angular.z
		
    def dbw_enabled_cb(self, msg):
        """
        Update the drive by wire state.

        Args:
            msg: message containing the drive by wire state.
        """

        # Extract dbw enabled state from Bool message
        self.dbw_enabled = msg.data

    def publish(self, throttle, brake, steering):
        """
        Publish throttle, steering, and brake values.

        Args:
            throttle: throttle value between 0 and 1
            steering: steering angle in radians
            brake: brake torque in N-m
        """  

        # Construct a new throttle command message
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle

        # Publish the throttle update
        self.throttle_pub.publish(tcmd)

        # Construct a new steering wheel angle command
        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steering

        # Publish the new steering wheel angle
        self.steer_pub.publish(scmd)
       
        # Construct a new brake torque torque command
        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake

        # Publish the new brake torque
        self.brake_pub.publish(bcmd)

if __name__ == '__main__':

    try:

        DBWNode()

    except rospy.ROSInterruptException:

        rospy.logerr('Could not start DBW node.')
    
