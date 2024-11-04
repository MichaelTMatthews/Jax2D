The scene functions provide utilities to change a particular scene (represented by a `SimState` object) in a variety of ways, including adding a thruster, adding revolute or fixed joints, or adding shapes.

# Creating a Scene
To create a scene, you can use the following code block

```python
static_sim_params = StaticSimParams()
sim_params = SimParams()
engine = PhysicsEngine(static_sim_params)

# Create scene
sim_state = create_empty_sim(static_sim_params, floor_offset=0.0)

```

# Editing a Scene
While you can edit a scene manually by changing the parameters, we recommend using the functions provided in `jax2d.scene` to edit a state.

!!! danger "Environment Size"
    Note, if the environment state has the maximum number of a certain entity type (e.g. polygons) and you try to add another one, it will result in a no-op.


::: jax2d.scene
