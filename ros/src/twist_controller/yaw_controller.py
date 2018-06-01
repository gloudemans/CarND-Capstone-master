from math import atan

class YawController(object):
    """
    Let a car with wheelbase w have its front wheels turned at angle theta. The front
    wheels travel along a circle with radius r1 and the rear wheels travel along a smaller
    circle with radius r0. If we make a simple sketch we can see the following relationships:

    * tan(theta) = w/r0
    * sin(theta) = w/r1
    * cos(theta) = r0/r1

    Given a desired rate of turn and the vehicle longitudinal velocity, the yaw controller
    should compute the required steering angle. The yaw controller should constrain the steering
    angle such that it remains within mechanical limits and such that lateral acceleration limits
    are not exceeded. When the vehcile is stopped, the controller should return a steering 
    angle of zero because no rate of turn can be achieved.    
    """

    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        """
        Args:
            wheel_base (double): turning radius in meters.
            steer_ratio (double): steering ratio (steering wheel angle per front wheel angle)
            max_lat_accel - maximum lateral acceleration in meters per second squared
            max_steer_angle - maximum steering wheel angle in radians
        """

        # Wheel base in meters
        self.wheel_base = wheel_base

        # Ratio of steering wheel angle change to front wheel angle change
        self.steer_ratio = steer_ratio

        # Set maximum lateral acceleration in meters per second squared
        self.max_lat_accel = max_lat_accel

        # Set maximum allowed steering angle in radians
        self.max_steer_angle = max_steer_angle

    def get_steering(self, longitudinal_velocity, angular_velocity):
        """Get steering wheel angle to produce the desired angular velocity
        subject to various limit checks. Longitudinal velocity is the component of
        velocity along the vehicle longitudinal axis as would be reported by 
        rotation of the rear wheels (assuming the car is wafer thin).

        Args:
            longitudinal_velocity (double): vehicle longitudinal velocity in meters per second
            angular_velocity (double): desired angular velocity in radians per second

        Returns:
            (double): steering wheel angle
        """

        # If the car is moving...
        if abs(longitudinal_velocity) > 0.0:

            # Compute yaw rate at which maximum lateral acceleration would be reached
            max_yaw_rate = abs(self.max_lat_accel / longitudinal_velocity);

            # Limit angular velocity so as not to exceed lateral acceleration limit
            angular_velocity = max(-max_yaw_rate, min(max_yaw_rate, angular_velocity))

            # Compute the required steering wheel angle
            steering_wheel_angle = self.steer_ratio * atan(self.wheel_base * angular_velocity / longitudinal_velocity)

            # Apply mechanical steering limits
            steering_wheel_angle = max(-self.max_steer_angle, min(self.max_steer_angle, steering_wheel_angle))

        else:

            # Can't steer when stopped
            steering_wheel_angle = 0.0

        # Return the final steering angle
        return steering_wheel_angle

