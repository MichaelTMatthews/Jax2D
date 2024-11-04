from dataclasses import field
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RigidBody:
    """This class represents a rigid body in the simulation. It can be a polygon or a circle.

    Parameters:
        position (jnp.ndarray): Position of the centroid of the shape in world space.

        rotation (float): Rotation of the RigidBody in radians

        velocity (jnp.ndarray): Velocity (x, y) in m/s

        angular_velocity (float): Angular velocity in radians/s

        inverse_mass (float): Inverse Mass, where 0 denotes an object with infinite mass (constant velocity)

        inverse_inertia (float): Inverse Inertia, where 0 denotes an object with infinite inertia (constant angular velocity)

        friction (float): Friction of the object

        restitution (float):How 'bouncy' the shape is, 0.0 is not bouncy at all and 1.0 is very bouncy. Restitution is calculated by multiplying the restitution values of two colliding shapes, so both have to have values > 0 to bounce. Due to baumgarte, the actual restitution is a bit higher, so setting this to 1 will cause energy to be created on collision

        collision_mode (int): 0 == doesn't collide with 1's. 1 = normal, i.e., it collides. 2 == collides with everything (including 0's).

        active (bool): Whether or not the shape is active

        # Polygon
        n_vertices (int): Must be >= 3, denoting how many vertices the polygon has
        vertices (jnp.ndarray): The vertices of the polygon in local space. The order must be clockwise, and the first vertex must be the top-left one.

        # Circle
        radius (float): If the shape is a circle, this is the radius

    """

    position: jnp.ndarray

    rotation: float

    velocity: jnp.ndarray

    angular_velocity: float

    inverse_mass: float

    inverse_inertia: float

    friction: float

    restitution: float

    collision_mode: int
    active: bool

    # Polygon
    n_vertices: int
    vertices: jnp.ndarray

    # Circle
    radius: float


@struct.dataclass
class CollisionManifold:
    normal: jnp.ndarray
    penetration: float
    collision_point: jnp.ndarray
    active: bool

    # Accumulated impulses
    acc_impulse_normal: float
    acc_impulse_tangent: float

    # Set at collision time to 'remember' the correct amount of bounce
    restitution_velocity_target: jnp.ndarray


@struct.dataclass
class Joint:
    """
    Represents a joint that connects two entities.

    Parameters:
        a_index (int): Global index of the first entity connected by the joint.
        b_index (int): Global index of the second entity connected by the joint.
        a_relative_pos (jnp.ndarray): Relative position of the joint on the first entity.
        b_relative_pos (jnp.ndarray): Relative position of the joint on the second entity.
        global_position (jnp.ndarray): Cached global position of the joint in world space.
        active (bool): Indicates whether the joint is currently active.

        acc_impulse (jnp.ndarray): Accumulated linear impulse for the joint.
        acc_r_impulse (jnp.ndarray): Accumulated rotational impulse for the joint.

        motor_speed (float): Speed of the motor attached to the joint, if any.
        motor_power (float): Maximum power output of the motor.
        motor_on (bool): Indicates whether the motor is currently active.

        motor_has_joint_limits (bool): Whether the motor has rotation limits for a revolute joint.
        min_rotation (float): Minimum rotation angle allowed for the joint in radians.
        max_rotation (float): Maximum rotation angle allowed for the joint in radians.

        is_fixed_joint (bool): Indicates if the joint is fixed, preventing movement.
        rotation (float): Fixed rotation angle if the joint is configured as a fixed joint.
    """

    a_index: int
    b_index: int
    a_relative_pos: jnp.ndarray
    b_relative_pos: jnp.ndarray
    global_position: jnp.ndarray  # Cached
    active: bool

    # Accumulated impulses
    acc_impulse: jnp.ndarray
    acc_r_impulse: jnp.ndarray

    # Motor
    motor_speed: float
    motor_power: float
    motor_on: bool

    # Revolute Joint
    motor_has_joint_limits: bool
    min_rotation: float
    max_rotation: float

    # Fixed joint
    is_fixed_joint: bool
    rotation: float


@struct.dataclass
class Thruster:
    """
    Represents a thruster attached to an object, which can apply a force to the shape.

    Parameters:
        object_index (int): Global index of the object to which the thruster is attached.
        relative_position (jnp.ndarray): Relative position of the thruster on the object.
        rotation (float): Rotation angle of the thruster in radians.
        power (float): Thrust power output of the thruster.
        global_position (jnp.ndarray): Cached global position of the thruster in world space.
        active (jnp.ndarray): Indicates whether the thruster is currently active.
    """

    object_index: int
    relative_position: jnp.ndarray
    rotation: float
    power: float
    global_position: jnp.ndarray  # Cached
    active: jnp.ndarray


@struct.dataclass
class SimState:
    polygon: RigidBody
    circle: RigidBody
    joint: Joint
    thruster: Thruster
    collision_matrix: jnp.ndarray

    # Impulse accumulation
    acc_rr_manifolds: CollisionManifold
    acc_cr_manifolds: CollisionManifold
    acc_cc_manifolds: CollisionManifold

    # Defaults
    gravity: jnp.ndarray


@struct.dataclass
class SimParams:
    # Timestep size
    dt: float = 1 / 60

    # Collision and joint coefficients
    slop: float = 0.01
    baumgarte_coefficient_joints_v: float = 2.0
    baumgarte_coefficient_joints_p: float = 0.7
    baumgarte_coefficient_fjoint_av: float = 2.0
    baumgarte_coefficient_rjoint_limit_av: float = 5.0
    baumgarte_coefficient_collision: float = 0.2
    joint_stiffness: float = 0.6

    # State clipping
    clip_position: float = 15
    clip_velocity: float = 100
    clip_angular_velocity: float = 50

    # Motors and thrusters
    base_motor_speed: float = 6.0  # rad/s
    base_motor_power: float = 900.0
    base_thruster_power: float = 10.0
    motor_decay_coefficient: float = 0.1
    motor_joint_limit: float = 0.1  # rad

    # Other defaults
    base_friction: float = 0.4


@struct.dataclass
class StaticSimParams:
    # State size
    num_polygons: int = 12
    num_circles: int = 12
    num_joints: int = 12
    num_thrusters: int = 12
    max_polygon_vertices: int = 4

    # Compute amount
    num_solver_iterations: int = 10
    solver_batch_size: int = 16
    do_warm_starting: bool = True
    num_static_fixated_polys: int = 4
