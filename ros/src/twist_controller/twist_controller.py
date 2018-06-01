from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

# Gas density in kilograms per liter
GAS_DENSITY = 2.858

# Brake torque to apply when stopped in newton meters
STOPPED_BRAKE_TORQUE = 400

class Controller(object):
    """
    The twist controller receives inputs describing the proposed vehicle 
    linear velocity and angular velocity and the current linear velocity and 
    is responsible for sending throttle, brake, and steering commands
    to approximate the proposed state.

    The yaw controller outputs a steering angle as a function of commanded
    vehicle velocity and angular acceleration. The yaw controller imposes 
    steering limits based on the mechanical limits of the steering system 
    and to keep lateral acceleration within limits. The calculation is 
    generally straightforward based on the relevant vehicle parameters. 

    In theory, we'd like to have information about wheel torque versus
    throttle position, vehicle mass, incline, and drag versus velocity.
    Using this information we could model the physics to do a very good
    job of manipulating the throttle and brake to hit our targets.
    Unfortunately we don't have really any of this, so we're just going
    to slap in a crude controller and fiddle around empirically until 
    it more or less works.

    """

    def __init__(self, *args, **kwargs):
        """
        Construct a new Controller object using the specified physical
        parameters.
        """

        # Retrieve parameters
        self.vehicle_mass    = kwargs['vehicle_mass']
        self.fuel_capacity   = kwargs['fuel_capacity']
        self.brake_deadband  = kwargs['brake_deadband']
        self.decel_limit     = kwargs['decel_limit']
        self.accel_limit     = kwargs['accel_limit']
        self.wheel_radius    = kwargs['wheel_radius']
        self.wheel_base      = kwargs['wheel_base']
        self.steer_ratio     = kwargs['steer_ratio']
        self.max_lat_accel   = kwargs['max_lat_accel']
        self.max_steer_angle = kwargs['max_steer_angle']

        # Invalidate time
        self.time       = None

        # Construct yaw controller with appropriate vehicle parameters
        self.yaw_controller = YawController(
            self.wheel_base, self.steer_ratio,
            self.max_lat_accel, self.max_steer_angle)

        # Set PID throttle controller parameters
        kp = 0.3
        ki = 0.1
        kd = 0.0
        throttle_min = 0.0
        throttle_max = 0.2

        # Construct the throttle controller
        self.throttle_controller = PID(kp, ki, kd, throttle_min, throttle_max)

        # Set velocity lowpass filter parameters
        tau = 0.50  # cutoff freq
        ts  = 0.02  # sample time
        self.velocity_lowpass = LowPassFilter(tau, ts)

    def control(self, proposed_linear_velocity, proposed_angular_velocity, current_linear_velocity, dbw_enabled, time):
        """
        Given the proposed linear velocity, proposed angular velocity, 
        current linear velocity, drive by wire enable state, and time, 
        determine the appropriate throttle, brake, and steering values.

        Args:
            proposed_linear_velocity: proposed linear velocity (m/S)
            proposed_angular_velocity: proposed angular velocity (m/S)
            current_linear_velocity: current linear velocity (m/S)
            dbw_enabled: drive by wire enable state
            time: rospy time

        Returns:
            throttle: throttle (0 to 1)
            brake: brake torque (N-m)
            steering: steering angle in (radians)
        """

        # Clear control
        throttle = 0
        brake = 0
        steering = 0

        # If time is available and drive by wire is enabled...
        if (self.time != None) and dbw_enabled:

            # Update the steering command
            steering = self.yaw_controller.get_steering(current_linear_velocity, proposed_angular_velocity)

            # Compute elapsed time since prior sample
            sample_time = time - self.time

            # Compute filtered velocity error
            velocity_error = self.velocity_lowpass.filt(proposed_linear_velocity - current_linear_velocity)

            # Update the throttle
            throttle = self.throttle_controller.step(velocity_error, sample_time)

            # If the vehicle is actually stopped...
            if (proposed_linear_velocity < self.brake_deadband) and (current_linear_velocity < self.brake_deadband):

                # Apply significant brake to keep the vehicle from rolling
                brake = STOPPED_BRAKE_TORQUE

                # Reset the controller so that it will start from zero
                self.throttle_controller.reset()

                # Inhibit the throttle
                throttle = 0

            # If the throttle is low and the velocity error is less than zero...
            elif (throttle < 0.1) and (velocity_error < 0.0):

                # Reset the controller so that it will start from zero
                self.throttle_controller.reset()

                # Inhibit the throttle
                throttle = 0

                # Set time constant for reconciling the velocity error in seconds
                time_constant = 1

                # Compute deceleration subject to deceleration limit
                deceleration = min(abs(velocity_error/time_constant), abs(self.decel_limit))

                # Compute braking torque for this deceleration
                brake = deceleration * (self.vehicle_mass + self.fuel_capacity*GAS_DENSITY) * self.wheel_radius        

        else:

            # Reset the controller so that it will start from zero
            self.throttle_controller.reset()

        # Update time state
        self.time = time

        # Return throttle, brake, steer
        return throttle, brake, steering
