from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jax2d.engine import (
    calc_inverse_mass_circle,
    calc_inverse_inertia_circle,
    calc_inverse_mass_polygon,
    calc_inverse_inertia_polygon,
    calculate_collision_matrix,
)
from jax2d.sim_state import SimState, StaticSimParams


def add_thruster_to_scene(
    sim_state: SimState, object_index: int, relative_position: jnp.ndarray, rotation: float, power=1.0
) -> tuple[SimState, int]:
    """
    Adds a thruster to the object specified by `object_index`.

    Args:
        sim_state (SimState):
        object_index (int): The global object index, i.e., in the range [0, num_objects), where polygons are first, then circles.
        relative_position (jnp.ndarray): Position of the thruster relative to the center of mass of the object.
        rotation (float): The rotation of the thruster, note that it is recommended to have the thruster always point in the direction of the center of mass of the object.
        power (float, optional): How strong the thruster is. Defaults to 1.0.

    Returns:
        tuple[SimState, int]: (new_sim_state, thruster_index)
    """
    thruster_index = jnp.argmin(sim_state.thruster.active)
    can_add_thruster = jnp.logical_not(sim_state.thruster.active.all())

    new_sim_state = sim_state.replace(
        thruster=sim_state.thruster.replace(
            object_index=sim_state.thruster.object_index.at[thruster_index].set(object_index),
            relative_position=sim_state.thruster.relative_position.at[thruster_index].set(relative_position),
            rotation=sim_state.thruster.rotation.at[thruster_index].set(rotation),
            power=sim_state.thruster.power.at[thruster_index].set(power),
            active=sim_state.thruster.active.at[thruster_index].set(True),
        )
    )

    return (
        jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(can_add_thruster, x, y),
            new_sim_state,
            sim_state,
        ),
        thruster_index,
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_revolute_joint_to_scene(
    sim_state: SimState,
    static_sim_params: StaticSimParams,
    a_index: int,
    b_index: int,
    a_relative_pos: jnp.ndarray,
    b_relative_pos: jnp.ndarray,
    motor_on: bool = False,
    motor_speed: float = 1.0,
    motor_power: float = 1.0,
    has_joint_limits: bool = False,
    min_rotation: float = -np.pi,
    max_rotation: float = np.pi,
) -> tuple[SimState, int]:
    """Adds a revolute joint to the scene, connecting the two shapes given by `a_index` and `b_index`.

    Args:
        sim_state (SimState):
        static_sim_params (StaticSimParams):
        a_index (int): Global object index in range [0, num_shapes)
        b_index (int): Global object index in range [0, num_shapes)
        a_relative_pos (jnp.ndarray): The joint's position relative to the center of mass of object `a_index`.
        b_relative_pos (jnp.ndarray): The joint's position relative to the center of mass of object `b_index`.
        motor_on (bool, optional): If this is true, the joint has a motor that can be activated by actions, applying a torque to these two objects. Defaults to False.
        motor_speed (float, optional): Only in use if `motor_on` is `True`, how fast can the motor rotate. Defaults to 1.0.
        motor_power (float, optional): Only in use if `motor_on` is `True`, the maximum torque the motor can apply. Defaults to 1.0.
        has_joint_limits (bool, optional): If true, then the joint has limits that it cannot exceed. Defaults to False.
        min_rotation (float, optional): Minimum joint limit. Defaults to -np.pi.
        max_rotation (float, optional): Maximum joint limit. Defaults to np.pi.

    Returns:
        tuple[SimState, int]: (new_sim_state, joint_index)
    """
    joint_index = jnp.argmin(sim_state.joint.active)
    can_add_joint = jnp.logical_not(sim_state.joint.active.all())

    new_sim_state = sim_state.replace(
        joint=sim_state.joint.replace(
            a_index=sim_state.joint.a_index.at[joint_index].set(a_index),
            b_index=sim_state.joint.b_index.at[joint_index].set(b_index),
            a_relative_pos=sim_state.joint.a_relative_pos.at[joint_index].set(a_relative_pos),
            b_relative_pos=sim_state.joint.b_relative_pos.at[joint_index].set(b_relative_pos),
            active=sim_state.joint.active.at[joint_index].set(True),
            is_fixed_joint=sim_state.joint.is_fixed_joint.at[joint_index].set(False),
            motor_on=sim_state.joint.motor_on.at[joint_index].set(motor_on),
            motor_speed=sim_state.joint.motor_speed.at[joint_index].set(motor_speed),
            motor_power=sim_state.joint.motor_power.at[joint_index].set(motor_power),
            motor_has_joint_limits=sim_state.joint.motor_has_joint_limits.at[joint_index].set(has_joint_limits),
            min_rotation=sim_state.joint.min_rotation.at[joint_index].set(min_rotation),
            max_rotation=sim_state.joint.max_rotation.at[joint_index].set(max_rotation),
        )
    )

    new_sim_state = new_sim_state.replace(
        collision_matrix=calculate_collision_matrix(static_sim_params, new_sim_state.joint)
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_joint, x, y), new_sim_state, sim_state),
        joint_index,
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_fixed_joint_to_scene(
    sim_state: SimState,
    static_sim_params: StaticSimParams,
    a_index: int,
    b_index: int,
    a_relative_pos: jnp.ndarray,
    b_relative_pos: jnp.ndarray,
) -> tuple[SimState, int]:
    """Adds a fixed joint to a scene, where a fixed joint does not allow the relative rotation between the two objects to change.

    Args:
        sim_state (SimState):
        static_sim_params (StaticSimParams):
        a_index (int): Global object index in range [0, num_shapes)
        b_index (int): Global object index in range [0, num_shapes)
        a_relative_pos (jnp.ndarray): The joint's position relative to the center of mass of object `a_index`.
        b_relative_pos (jnp.ndarray): The joint's position relative to the center of mass of object `b_index`.

    Returns:
        tuple[SimState, int]: (new_sim_state, joint_index)
    """
    joint_index = jnp.argmin(sim_state.joint.active)
    can_add_joint = jnp.logical_not(sim_state.joint.active.all())

    new_sim_state = sim_state.replace(
        joint=sim_state.joint.replace(
            a_index=sim_state.joint.a_index.at[joint_index].set(a_index),
            b_index=sim_state.joint.b_index.at[joint_index].set(b_index),
            a_relative_pos=sim_state.joint.a_relative_pos.at[joint_index].set(a_relative_pos),
            b_relative_pos=sim_state.joint.b_relative_pos.at[joint_index].set(b_relative_pos),
            active=sim_state.joint.active.at[joint_index].set(True),
            is_fixed_joint=sim_state.joint.is_fixed_joint.at[joint_index].set(True),
        )
    )

    new_sim_state = new_sim_state.replace(
        collision_matrix=calculate_collision_matrix(static_sim_params, new_sim_state.joint)
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_joint, x, y), new_sim_state, sim_state),
        joint_index,
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_circle_to_scene(
    sim_state: SimState,
    static_sim_params: StaticSimParams,
    position: jnp.ndarray,
    radius: float,
    rotation: float = 0.0,
    velocity: jnp.ndarray = jnp.zeros(2),
    angular_velocity: float = 0.0,
    density: float = 1.0,
    friction: float = 1.0,
    restitution: float = 0.0,
    fixated: bool = False,
) -> tuple[SimState, tuple[int, int]]:
    """Adds a circle to the scene, with the properties specified by the given parameters.

    Args:
        sim_state (SimState):
        static_sim_params (StaticSimParams):
        position (jnp.ndarray): Position of the center in world space
        radius (float): Radius of the circle
        rotation (float, optional): Rotation. Defaults to 0.0.
        velocity (jnp.ndarray, optional): Initial velocity. Defaults to jnp.zeros(2).
        angular_velocity (float, optional): Initial angular velocity. Defaults to 0.0.
        density (float, optional): Shape's density. Defaults to 1.0.
        friction (float, optional): Friction coefficient. Defaults to 1.0.
        restitution (float, optional): How 'bouncy' the shape is, 0.0 is not bouncy at all and 1.0 is very bouncy. Restitution is calculated by multiplying the restitution values of two colliding shapes, so both have to have values > 0 to bounce. Defaults to 0.0.
        fixated (bool, optional): If true, the shape has infinite mass, meaning that it will not move and acts as a fixed shape. Defaults to False.

    Returns:
        tuple[SimState, tuple[int, int]]: (new_sim_state, (circle_index, global_index)): The two indices we return are: (a) the index of the newly-added circle in the state.circle array, and (b) the global index, in the range [0, num_shapes), where polygons are first, then circles.
    """
    circle_index = jnp.argmin(sim_state.circle.active)
    can_add_circle = jnp.logical_not(sim_state.circle.active.all())

    inverse_mass = calc_inverse_mass_circle(radius, density)
    inverse_inertia = calc_inverse_inertia_circle(radius, density)

    inverse_mass *= jnp.logical_not(fixated)
    inverse_inertia *= jnp.logical_not(fixated)

    new_sim_state = sim_state.replace(
        circle=sim_state.circle.replace(
            position=sim_state.circle.position.at[circle_index].set(position),
            radius=sim_state.circle.radius.at[circle_index].set(radius),
            rotation=sim_state.circle.rotation.at[circle_index].set(rotation),
            velocity=sim_state.circle.velocity.at[circle_index].set(velocity),
            angular_velocity=sim_state.circle.angular_velocity.at[circle_index].set(angular_velocity),
            friction=sim_state.circle.friction.at[circle_index].set(friction),
            restitution=sim_state.circle.restitution.at[circle_index].set(restitution),
            inverse_mass=sim_state.circle.inverse_mass.at[circle_index].set(inverse_mass),
            inverse_inertia=sim_state.circle.inverse_inertia.at[circle_index].set(inverse_inertia),
            active=sim_state.circle.active.at[circle_index].set(True),
        )
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_circle, x, y), new_sim_state, sim_state),
        (
            circle_index,
            circle_index + static_sim_params.num_polygons,
        ),
    )


@partial(jax.jit, static_argnames="static_sim_params")
def add_rectangle_to_scene(
    sim_state: SimState,
    static_sim_params: StaticSimParams,
    position: jnp.ndarray,
    dimensions: jnp.ndarray,
    rotation: float = 0.0,
    velocity: jnp.ndarray = jnp.zeros(2),
    angular_velocity: float = 0.0,
    density: float = 1.0,
    friction: float = 1.0,
    restitution: float = 0.0,
    fixated: bool = False,
) -> tuple[SimState, tuple[int, int]]:
    """Adds a rectangle to the scene.

    Args:
        sim_state (SimState):
        static_sim_params (StaticSimParams):
        position (jnp.ndarray): Position of the center of the rectangle in world space
        dimensions (jnp.ndarray): (width, height) of the rectangle
        rotation (float, optional): Rotation. Defaults to 0.0.
        velocity (jnp.ndarray, optional): Initial velocity. Defaults to jnp.zeros(2).
        angular_velocity (float, optional): Initial angular velocity. Defaults to 0.0.
        density (float, optional): Shape's density. Defaults to 1.0.
        friction (float, optional): Friction coefficient. Defaults to 1.0.
        restitution (float, optional): How 'bouncy' the shape is, 0.0 is not bouncy at all and 1.0 is very bouncy. Restitution is calculated by multiplying the restitution values of two colliding shapes, so both have to have values > 0 to bounce. Defaults to 0.0.
        fixated (bool, optional): If true, the shape has infinite mass, meaning that it will not move and acts as a fixed shape. Defaults to False.

    Returns:
        tuple[SimState, tuple[int, int]]: (new_sim_state, (polygon_index, global_index)): The two indices we return are: (a) the index of the newly-added polygon in the state.polygon array, and (b) the global index, in the range [0, num_shapes), where polygons are first, then circles.
    """
    half_dims = dimensions / 2.0
    vertices = jnp.array(
        [
            [-half_dims[0], half_dims[1]],
            [half_dims[0], half_dims[1]],
            [half_dims[0], -half_dims[1]],
            [-half_dims[0], -half_dims[1]],
        ]
    )

    return add_polygon_to_scene(
        sim_state,
        static_sim_params,
        position,
        vertices,
        4,
        rotation,
        velocity,
        angular_velocity,
        density,
        friction,
        restitution,
        fixated,
    )


@partial(jax.jit, static_argnames=["static_sim_params", "n_vertices"])
def add_polygon_to_scene(
    sim_state: SimState,
    static_sim_params: StaticSimParams,
    position: jnp.ndarray,
    vertices: jnp.ndarray,
    n_vertices: int,
    rotation: float = 0.0,
    velocity: jnp.ndarray = jnp.zeros(2),
    angular_velocity: float = 0.0,
    density: float = 1.0,
    friction: float = 1.0,
    restitution: float = 0.0,
    fixated: bool = False,
) -> tuple[SimState, tuple[int, int]]:
    """Adds a polygon to the scene.

    Args:
        sim_state (SimState):
        static_sim_params (StaticSimParams):
        position (jnp.ndarray): Position of the center of the polygon in world space.
        vertices (jnp.ndarray): A list of vertices, which has shape (n_vertices, 2). The ordering of these vertices is important, and has to be clockwise, starting from the top-left vertex.
        n_vertices (int): How many vertices are in the polygon.
        rotation (float, optional): Rotation. Defaults to 0.0.
        velocity (jnp.ndarray, optional): Initial velocity. Defaults to jnp.zeros(2).
        angular_velocity (float, optional): Initial angular velocity. Defaults to 0.0.
        density (float, optional): Shape's density. Defaults to 1.0.
        friction (float, optional): Friction coefficient. Defaults to 1.0.
        restitution (float, optional): How 'bouncy' the shape is, 0.0 is not bouncy at all and 1.0 is very bouncy. Restitution is calculated by multiplying the restitution values of two colliding shapes, so both have to have values > 0 to bounce. Defaults to 0.0.
        fixated (bool, optional): If true, the shape has infinite mass, meaning that it will not move and acts as a fixed shape. Defaults to False.

    Returns:
        tuple[SimState, tuple[int, int]]: (new_sim_state, (polygon_index, global_index)): The two indices we return are: (a) the index of the newly-added polygon in the state.polygon array, and (b) the global index, in the range [0, num_shapes), where polygons are first, then circles.
    """
    # Fill vertices up to maxP
    vertices = jnp.zeros((static_sim_params.max_polygon_vertices, 2)).at[:n_vertices].set(vertices)

    polygon_index = jnp.argmin(sim_state.polygon.active)
    can_add_polygon = jnp.logical_not(sim_state.polygon.active.all())

    # Adjust position and vertices according to new CoM
    # This will only be non-zero if the current CoM is wrong
    inverse_mass, delta_centre_of_mass = calc_inverse_mass_polygon(vertices, n_vertices, static_sim_params, density)
    position += delta_centre_of_mass
    vertices -= delta_centre_of_mass[None, :]

    inverse_inertia = calc_inverse_inertia_polygon(vertices, n_vertices, static_sim_params, density)

    inverse_mass *= jnp.logical_not(fixated)
    inverse_inertia *= jnp.logical_not(fixated)

    new_sim_state = sim_state.replace(
        polygon=sim_state.polygon.replace(
            position=sim_state.polygon.position.at[polygon_index].set(position),
            vertices=sim_state.polygon.vertices.at[polygon_index].set(vertices),
            rotation=sim_state.polygon.rotation.at[polygon_index].set(rotation),
            velocity=sim_state.polygon.velocity.at[polygon_index].set(velocity),
            angular_velocity=sim_state.polygon.angular_velocity.at[polygon_index].set(angular_velocity),
            friction=sim_state.polygon.friction.at[polygon_index].set(friction),
            restitution=sim_state.polygon.restitution.at[polygon_index].set(restitution),
            inverse_mass=sim_state.polygon.inverse_mass.at[polygon_index].set(inverse_mass),
            inverse_inertia=sim_state.polygon.inverse_inertia.at[polygon_index].set(inverse_inertia),
            active=sim_state.polygon.active.at[polygon_index].set(True),
            n_vertices=sim_state.polygon.n_vertices.at[polygon_index].set(n_vertices),
        )
    )

    return (
        jax.tree_util.tree_map(lambda x, y: jax.lax.select(can_add_polygon, x, y), new_sim_state, sim_state),
        (polygon_index, polygon_index),
    )
