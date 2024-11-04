Below shows an example of how to use Jax2D to create and run a scene.  For the full code see [examples/car.py](examples/car.py). This produces the following scene (rendered with [JaxGL](https://github.com/FLAIROx/JaxGL))

<p align="center">
 <img width="50%" src="../car.gif" />
</p>

```python
# Create engine with default parameters
static_sim_params = StaticSimParams()
sim_params = SimParams()
engine = PhysicsEngine(static_sim_params)

# Create scene
sim_state = create_empty_sim(static_sim_params, floor_offset=0.0)

# Create a rectangle for the car body
sim_state, (_, r_index) = add_rectangle_to_scene(
    sim_state, static_sim_params, position=jnp.array([2.0, 1.0]),
    dimensions=jnp.array([1.0, 0.4])
)

# Create circles for the wheels of the car
sim_state, (_, c1_index) = add_circle_to_scene(
    sim_state, static_sim_params, position=jnp.array([1.5, 1.0]), radius=0.35
)
sim_state, (_, c2_index) = add_circle_to_scene(
    sim_state, static_sim_params, position=jnp.array([2.5, 1.0]), radius=0.35
)

# Join the wheels to the car body with revolute joints
# Relative positions are from the centre of masses of each object
sim_state, _ = add_revolute_joint_to_scene(
    sim_state,
    static_sim_params,
    a_index=r_index,
    b_index=c1_index,
    a_relative_pos=jnp.array([-0.5, 0.0]),
    b_relative_pos=jnp.zeros(2),
    motor_on=True,
)
sim_state, _ = add_revolute_joint_to_scene(
    sim_state,
    static_sim_params,
    a_index=r_index,
    b_index=c2_index,
    a_relative_pos=jnp.array([0.5, 0.0]),
    b_relative_pos=jnp.zeros(2),
    motor_on=True,
)

# Add a triangle for a ramp - we fixate the ramp so it can't move
triangle_vertices = jnp.array([[0.5, 0.1], [0.5, -0.1], [-0.5, -0.1]])
sim_state, _ = add_polygon_to_scene(
    sim_state,
    static_sim_params,
    position=jnp.array([2.7, 0.1]),
    vertices=triangle_vertices,
    n_vertices=3,
    fixated=True,
)


# Run scene
step_fn = jax.jit(engine.step)

while True:
    # We activate all motors and thrusters
    actions = jnp.ones(static_sim_params.num_joints + static_sim_params.num_thrusters)
    sim_state, _ = step_fn(sim_state, sim_params, actions)
    
    # Do rendering...
```