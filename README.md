# Jax2D
<p align="center">
        <a href= "https://pypi.org/project/jax2d/">
        <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" /></a>
        <a href= "https://pypi.org/project/jax2d/">
        <img src="https://img.shields.io/badge/pypi-1.0.0-green" /></a>
       <a href= "https://github.com/MichaelTMatthews/Craftax/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-yellow" /></a>
       <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>

Jax2D is a 2D rigid-body physics engine written entirely in [JAX](https://github.com/google/jax) and based off the [Box2D](https://github.com/erincatto/box2d) engine.
Unlike other JAX physics engines, Jax2D is dynamic with respect to scene configuration, allowing heterogeneous scenes to be parallelised with `vmap`.
Jax2D was initially created for the backend of the [Kinetix](https://github.com/FLAIROx/Kinetix) project and was developed by Michael_{[Matthews](https://github.com/MichaelTMatthews), [Beukman](https://github.com/Michael-Beukman)}.

<p align="center">
 <img width="50%" src="images/tower.gif" />
</p>

# Why should I use Jax2D?
The main reason to use Jax2D over other JAX physics engines such as [Brax](https://github.com/google/brax) or [MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx) is that Jax2D scenes are (largely) dynamically specified.
Jax2D always has O(n^2) runtime with respect to the number of entities in a scene, since we must always calculate the full collision resolution for every pair of entities.
This means it is usually not appropriate for simulating scenes with large numbers (>100) of entities.

In short: Jax2D excels at simulating **lots** of **small** and **diverse** scenes in parallel **very fast**.

# Example Usage
Below shows an example of how to use Jax2D to create and run a scene.  For the full code see [examples/car.py](examples/car.py)
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
This produces the following scene (rendered with [JaxGL](https://github.com/FLAIROx/JaxGL))

<p align="center">
 <img width="50%" src="images/car.gif" />
</p>

# More Complex Levels
For creating and using more complicated levels, we recommend using the built-in editors provided in [Kinetix](https://github.com/FLAIROx/Kinetix) (or the online version available [here](https://kinetix-env.github.io/gallery.html?editor=true)).

# Installation
To use Jax2D in your work you can install via PyPi:
```commandline
pip install jax2d
```

If you want to extend Jax2D you can install as follows:
```commandline
git clone https://github.com/MichaelTMatthews/Jax2D
cd Jax2D
pip install -e ".[dev]"
pre-commit install
```

# See Also
- 🍎 [Box2D](https://github.com/erincatto/box2d) The original C physics engine
- 🤖 [Kinetix](https://github.com/FLAIROx/Kinetix) Jax2D as a reinforcement learning environment
- 🌐 [Kinetix.js](https://github.com/Michael-Beukman/Kinetix.js) Jax2D reimplemented in Javascript, with a live demo [here](https://kinetix-env.github.io/gallery.html?editor=true).
- 🦾 [Brax](https://github.com/google/brax) 3D physics in JAX
- 🦿 [MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx) MuJoCo in JAX
- 👨‍💻 [JaxGL](https://github.com/FLAIROx/JaxGL) Rendering in JAX

# Citation
If you use Jax2D in your work please cite it as follows:
```
@article{matthews2024kinetix,
      title={Kinetix: Investigating the Training of General Agents through Open-Ended Physics-Based Control Tasks}, 
      author={Michael Matthews and Michael Beukman and Chris Lu and Jakob Foerster},
      year={2024},
      eprint={2410.23208},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.23208}, 
}
```

# Acknowledgements
We would like to thank [Erin Catto](https://github.com/erincatto) and [Randy Gaul](https://randygaul.github.io/) for their invaluable online materials that allowed the creation of this engine.  If you would like to develop your own physics engine, we recommend starting [here](https://randygaul.github.io/collision-detection/2019/06/19/Collision-Detection-in-2D-Some-Steps-for-Success.html).
