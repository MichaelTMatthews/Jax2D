import math
from functools import partial

import chex
import jax
import jax.numpy as jnp

from jax2d.collision import (
    generate_manifold_circle_circle,
    resolve_collision,
    resolve_warm_starting_impulse,
    generate_manifolds_polygon_polygon,
    generate_manifold_circle_polygon,
)
from jax2d.sim_state import (
    SimState,
    RigidBody,
    Joint,
    CollisionManifold,
    Thruster,
    SimParams,
    StaticSimParams,
)
from jax2d.joint import (
    apply_motor,
    resolve_joint_warm_start,
    resolve_joint,
)
from jax2d.maths import rmat, zero_to_one


def recompute_global_joint_positions(state: SimState, static_params: StaticSimParams):
    jshapes = jax.vmap(select_shape, (None, 0, None))(state, state.joint.a_index, static_params)
    tshapes = jax.vmap(select_shape, (None, 0, None))(state, state.thruster.object_index, static_params)
    jpos = jshapes.position
    tpos = tshapes.position

    jmat = jax.vmap(rmat)(jshapes.rotation)
    tmat = jax.vmap(rmat)(tshapes.rotation)

    @jax.vmap
    def mm(a, b):
        return jnp.matmul(a, b)

    state = state.replace(
        joint=state.joint.replace(
            global_position=jpos + mm(jmat, state.joint.a_relative_pos),
        ),
        thruster=state.thruster.replace(
            global_position=tpos + mm(tmat, state.thruster.relative_position),
        ),
    )
    return state


def clip_state(state: SimState, params: SimParams):
    state = state.replace(
        circle=state.circle.replace(
            position=jnp.clip(
                state.circle.position,
                min=-params.clip_position,
                max=params.clip_position,
            ),
            velocity=jnp.clip(
                state.circle.velocity,
                min=-params.clip_velocity,
                max=params.clip_velocity,
            ),
            angular_velocity=jnp.clip(
                state.circle.angular_velocity,
                min=-params.clip_angular_velocity,
                max=params.clip_angular_velocity,
            ),
        ),
        polygon=state.polygon.replace(
            position=jnp.clip(
                state.polygon.position,
                min=-params.clip_position,
                max=params.clip_position,
            ),
            velocity=jnp.clip(
                state.polygon.velocity,
                min=-params.clip_velocity,
                max=params.clip_velocity,
            ),
            angular_velocity=jnp.clip(
                state.polygon.angular_velocity,
                min=-params.clip_angular_velocity,
                max=params.clip_angular_velocity,
            ),
        ),
    )
    return state


def calc_inverse_inertia_circle(radius, density):
    inertia = math.pi * jnp.pow(radius, 4) / 4 * density
    return 1 / zero_to_one(inertia)


def calc_inverse_mass_circle(radius, density):
    mass = radius * radius * math.pi * density
    return 1 / zero_to_one(mass)


def calc_inverse_inertia_polygon(vertices, n_vertices, static_params, density):
    def _calc_triangle_inertia(v):
        # Point 0 is (0,0) (i.e. the centroid)
        p1 = v[1]
        p2 = v[0]

        D = jnp.cross(p1, p2)
        intx2 = p1[0] * p1[0] + p2[0] * p1[0] + p2[0] * p2[0]
        inty2 = p1[1] * p1[1] + p2[1] * p1[1] + p2[1] * p2[1]

        I = (0.25 * D / 3.0) * (intx2 + inty2)

        return jnp.abs(I)

    t_indexes = jnp.concatenate(
        [
            (jnp.arange(static_params.max_polygon_vertices, dtype=jnp.int32))[None, :],
            (
                (jnp.arange(static_params.max_polygon_vertices, dtype=jnp.int32) + 1)
                % static_params.max_polygon_vertices
            )[None, :],
        ],
        axis=0,
    ).transpose((1, 0))
    t_vertices = vertices[t_indexes]

    t_mask = jnp.arange(static_params.max_polygon_vertices) < n_vertices
    t_inertias = jax.vmap(_calc_triangle_inertia)(t_vertices) * t_mask

    v = t_inertias.sum() * density
    return 1 / zero_to_one(v)


def calc_inverse_mass_polygon(vertices, n_vertices, static_params, density):
    def _calc_triangle_mass(v):
        width = jnp.linalg.norm(v[1] - v[0])
        t = (v[1] - v[0]) / jax.lax.select(width == 0, 1.0, width)
        a = v[0] + jnp.dot(v[2] - v[0], t) * t
        height = jnp.linalg.norm(v[2] - a)

        return width * height / 2.0

    n_triangles = static_params.max_polygon_vertices - 2
    t_indexes = jnp.concatenate(
        [
            jnp.zeros(n_triangles, dtype=jnp.int32)[None, :],
            (jnp.arange(n_triangles, dtype=jnp.int32) + 1)[None, :],
            (jnp.arange(n_triangles, dtype=jnp.int32) + 2)[None, :],
        ],
        axis=0,
    ).transpose((1, 0))
    t_vertices = vertices[t_indexes]

    t_mask = jnp.arange(n_triangles) < n_vertices - 2
    t_masses = jax.vmap(_calc_triangle_mass)(t_vertices) * t_mask

    v_mask = jnp.arange(static_params.max_polygon_vertices) < n_vertices
    com = (vertices * v_mask[:, None]).sum(axis=0) / n_vertices

    return 1 / zero_to_one(t_masses.sum() * density), com


def recalculate_mass_and_inertia(state: SimState, static_params, polygon_densities, circle_densities):
    # Note: We leave objects with 0 mass/inertia unchanged

    # We first calculate centroids for polygons and move them accordingly.
    # This is required to ensure that the centroid is inside the polygon for inertia calculations.
    polygon_inverse_mass, pos_diff = jax.vmap(calc_inverse_mass_polygon, in_axes=(0, 0, None, 0))(
        state.polygon.vertices,
        state.polygon.n_vertices,
        static_params,
        polygon_densities,
    )
    polygon_inverse_mass *= state.polygon.inverse_mass != 0
    pos_diff = jnp.where((state.polygon.inverse_mass == 0)[:, None], jnp.zeros_like(pos_diff), pos_diff)

    state = state.replace(
        polygon=state.polygon.replace(
            position=state.polygon.position + pos_diff,
            vertices=state.polygon.vertices - pos_diff[:, None, :],
        ),
    )

    polygon_inverse_inertia = jax.vmap(calc_inverse_inertia_polygon, in_axes=(0, 0, None, 0))(
        state.polygon.vertices,
        state.polygon.n_vertices,
        static_params,
        polygon_densities,
    ) * (state.polygon.inverse_inertia != 0)

    circle_inverse_mass = jax.vmap(calc_inverse_mass_circle, (0, 0))(state.circle.radius, circle_densities) * (
        state.circle.inverse_mass != 0
    )
    circle_inverse_inertia = jax.vmap(calc_inverse_inertia_circle, (0, 0))(state.circle.radius, circle_densities) * (
        state.circle.inverse_mass != 0
    )

    return state.replace(
        circle=state.circle.replace(
            inverse_mass=circle_inverse_mass,
            inverse_inertia=circle_inverse_inertia,
        ),
        polygon=state.polygon.replace(
            inverse_mass=polygon_inverse_mass,
            inverse_inertia=polygon_inverse_inertia,
        ),
    )


@partial(jax.jit, static_argnums=(0,))
def calculate_collision_matrix(static_sim_params, joints):
    #       Poly  Circle
    # Poly
    # Circle

    matrix_size = static_sim_params.num_polygons + static_sim_params.num_circles

    collision_matrix = jnp.logical_not(jnp.eye(matrix_size, dtype=bool))

    def make_joint_adder(joints):
        def _add_joint_to_matrix(collision_matrix, j_index):
            j = jax.tree.map(lambda x: x[j_index], joints)

            row = collision_matrix[j.a_index] & collision_matrix[j.b_index]
            col = collision_matrix[:, j.b_index] & collision_matrix[:, j.a_index]
            joint_collisions = row & col

            new_collision_matrix = collision_matrix.at[j.a_index].set(joint_collisions)
            new_collision_matrix = new_collision_matrix.at[:, j.b_index].set(joint_collisions)
            new_collision_matrix = new_collision_matrix.at[j.b_index].set(joint_collisions)
            new_collision_matrix = new_collision_matrix.at[:, j.a_index].set(joint_collisions)

            new_collision_matrix = jax.lax.select(j.active, new_collision_matrix, collision_matrix)

            return new_collision_matrix, None

        return _add_joint_to_matrix

    joint_adder = make_joint_adder(joints)

    collision_matrix, _ = jax.lax.scan(
        joint_adder,
        collision_matrix,
        jnp.tile(jnp.arange(static_sim_params.num_joints), static_sim_params.num_joints),
    )

    return collision_matrix


def select_shape(state, index, static_sim_params):
    # Used a unified indexed for [poly0, ..., polyN, circle0, ..., circleN]
    polygon = jax.tree.map(lambda x: x[index], state.polygon)
    circle = jax.tree.map(lambda x: x[index - static_sim_params.num_polygons], state.circle)

    return jax.tree.map(
        lambda r, c: jax.lax.select(index < static_sim_params.num_polygons, r, c),
        polygon,
        circle,
    )


def make_impulse_warm_starting_fn(static_sim_params, shape1_poly, shape2_poly, n_manifolds, index_pairs):
    both_poly = shape1_poly & shape2_poly

    def _apply_warm_starting(state, manifolds):
        def _apply_warm_starting_single_impulse(index):
            s1_index = index_pairs[index, 0]
            s2_index = index_pairs[index, 1]

            does_collide = state.collision_matrix[
                s1_index + static_sim_params.num_polygons * jnp.logical_not(shape1_poly),
                s2_index + static_sim_params.num_polygons * jnp.logical_not(shape2_poly),
            ]

            if shape1_poly:
                s1 = jax.tree.map(lambda x: x[s1_index], state.polygon)
            else:
                s1 = jax.tree.map(lambda x: x[s1_index], state.circle)

            if shape2_poly:
                s2 = jax.tree.map(lambda x: x[s2_index], state.polygon)
            else:
                s2 = jax.tree.map(lambda x: x[s2_index], state.circle)

            if shape1_poly:
                m1 = jax.tree.map(lambda x: x[index, 0], manifolds)
            else:
                m1 = jax.tree.map(lambda x: x[index], manifolds)

            s1_dv, s2_dv, s1_drv, s2_drv = resolve_warm_starting_impulse(s1, s2, m1, does_collide)

            if both_poly:
                m2 = jax.tree.map(lambda x: x[index, 1], manifolds)

                s1 = s1.replace(
                    velocity=s1.velocity + s1_dv,
                    angular_velocity=s1.angular_velocity + s1_drv,
                )
                s2 = s2.replace(
                    velocity=s2.velocity + s2_dv,
                    angular_velocity=s2.angular_velocity + s2_drv,
                )

                s1_dv2, s2_dv2, s1_drv2, s2_drv2 = resolve_warm_starting_impulse(s1, s2, m2, does_collide)
                s1_dv, s2_dv, s1_drv, s2_drv = (
                    s1_dv + s1_dv2,
                    s2_dv + s2_dv2,
                    s1_drv + s1_drv2,
                    s2_drv + s2_drv2,
                )

            return s1_dv, s2_dv, s1_drv, s2_drv

        s1_dv, s2_dv, s1_drv, s2_drv = jax.vmap(_apply_warm_starting_single_impulse)(jnp.arange(n_manifolds))

        rv = state.polygon.velocity
        rav = state.polygon.angular_velocity
        cv = state.circle.velocity
        cav = state.circle.angular_velocity

        if shape1_poly:
            rv = rv.at[index_pairs[:, 0]].add(s1_dv)
            rav = rav.at[index_pairs[:, 0]].add(s1_drv)
        else:
            cv = cv.at[index_pairs[:, 0]].add(s1_dv)
            cav = cav.at[index_pairs[:, 0]].add(s1_drv)

        if shape2_poly:
            rv = rv.at[index_pairs[:, 1]].add(s2_dv)
            rav = rav.at[index_pairs[:, 1]].add(s2_drv)
        else:
            cv = cv.at[index_pairs[:, 1]].add(s2_dv)
            cav = cav.at[index_pairs[:, 1]].add(s2_drv)

        state = state.replace(
            polygon=state.polygon.replace(
                velocity=rv,
                angular_velocity=rav,
            ),
            circle=state.circle.replace(
                velocity=cv,
                angular_velocity=cav,
            ),
        )

        return state

    return _apply_warm_starting


def make_impulse_resolver_fn(
    static_sim_params,
    shape1_poly,
    shape2_poly,
    n_manifolds,
    n_manifold_batches,
    batch_size,
    n_manifolds_to_compute,
    index_pairs,
):
    is_rr = shape1_poly & shape2_poly

    def _resolve_manifolds(state, manifolds, params):
        def _resolve_manifold_batch(carry, indexes):
            def _resolve_single_manifold(index, secondary_index):
                state, manifolds, secondary_index = carry

                # Obtain shapes
                s1_index = index_pairs[index, 0]
                s2_index = index_pairs[index, 1]

                does_collide = state.collision_matrix[
                    s1_index + static_sim_params.num_polygons * jnp.logical_not(shape1_poly),
                    s2_index + static_sim_params.num_polygons * jnp.logical_not(shape2_poly),
                ]

                if shape1_poly:
                    s1 = jax.tree.map(lambda x: x[s1_index], state.polygon)
                else:
                    s1 = jax.tree.map(lambda x: x[s1_index], state.circle)

                if shape2_poly:
                    s2 = jax.tree.map(lambda x: x[s2_index], state.polygon)
                else:
                    s2 = jax.tree.map(lambda x: x[s2_index], state.circle)

                # Obtain manifold
                if is_rr:
                    m1 = jax.tree.map(lambda x: x[index, secondary_index], manifolds)
                else:
                    m1 = jax.tree.map(lambda x: x[index], manifolds)

                # Resolve collision
                # s1_dv, s2_dv, s1_drv, s2_drv, iN, iT
                res = resolve_collision(s1, s2, m1, does_collide, params)
                res = jax.tree.map(lambda x: jax.lax.select(index == -1, jnp.zeros_like(x), x), res)

                return res

            state, manifolds, secondary_index = carry

            s1_dv, s2_dv, s1_drv, s2_drv, iN, iT = jax.vmap(_resolve_single_manifold, in_axes=(0, None))(
                indexes, secondary_index
            )

            # Accumulate impulses into collision manifold
            if is_rr:
                manifolds = manifolds.replace(
                    acc_impulse_normal=manifolds.acc_impulse_normal.at[indexes, secondary_index].set(iN),
                    acc_impulse_tangent=manifolds.acc_impulse_tangent.at[indexes, secondary_index].set(iT),
                )
            else:
                manifolds = manifolds.replace(
                    acc_impulse_normal=manifolds.acc_impulse_normal.at[indexes].set(iN),
                    acc_impulse_tangent=manifolds.acc_impulse_tangent.at[indexes].set(iT),
                )

            rv = state.polygon.velocity
            rav = state.polygon.angular_velocity
            cv = state.circle.velocity
            cav = state.circle.angular_velocity

            if shape1_poly:
                rv = rv.at[index_pairs[indexes, 0]].add(s1_dv)
                rav = rav.at[index_pairs[indexes, 0]].add(s1_drv)
            else:
                cv = cv.at[index_pairs[indexes, 0]].add(s1_dv)
                cav = cav.at[index_pairs[indexes, 0]].add(s1_drv)

            if shape2_poly:
                rv = rv.at[index_pairs[indexes, 1]].add(s2_dv)
                rav = rav.at[index_pairs[indexes, 1]].add(s2_drv)
            else:
                cv = cv.at[index_pairs[indexes, 1]].add(s2_dv)
                cav = cav.at[index_pairs[indexes, 1]].add(s2_drv)

            state = state.replace(
                polygon=state.polygon.replace(
                    velocity=rv,
                    angular_velocity=rav,
                ),
                circle=state.circle.replace(
                    velocity=cv,
                    angular_velocity=cav,
                ),
            )

            return (state, manifolds, secondary_index), None

        # Spread out the active manifolds as much as possible
        active = manifolds.active
        if is_rr:
            active = jnp.logical_or(active[:, 0], active[:, 1])

        ordering = active.argsort()
        indexes = -jnp.ones(n_manifolds_to_compute, dtype=jnp.int32)
        indexes = indexes.at[jnp.arange(n_manifolds)].set(ordering)
        indexes = indexes.reshape((batch_size, n_manifold_batches)).T

        (state, manifolds, _), _ = jax.lax.scan(_resolve_manifold_batch, (state, manifolds, 0), xs=indexes)

        if is_rr:
            (state, manifolds, _), _ = jax.lax.scan(_resolve_manifold_batch, (state, manifolds, 1), xs=indexes)

        return state, manifolds

    return _resolve_manifolds


def make_joint_warm_starting_fn(static_sim_params, n_joints):
    def _resolve_joint_warm_starting(state):
        # Revolute Joints
        def _apply_single_joint_ws(joint_index):
            joint = jax.tree.map(lambda x: x[joint_index], state.joint)

            r1 = select_shape(state, joint.a_index, static_sim_params)
            r2 = select_shape(state, joint.b_index, static_sim_params)

            a_dv, b_dv, a_drv, b_drv = resolve_joint_warm_start(r1, r2, joint)

            return joint.a_index, joint.b_index, a_dv, b_dv, a_drv, b_drv

        ai, bi, a_dv, b_dv, a_drv, b_drv = jax.vmap(_apply_single_joint_ws)(jnp.arange(n_joints))

        poly_ai = ai < static_sim_params.num_polygons
        poly_bi = bi < static_sim_params.num_polygons

        # Poly
        new_poly_vel = state.polygon.velocity.at[ai].add(a_dv * poly_ai[:, None])
        new_poly_vel = new_poly_vel.at[bi].add(b_dv * poly_bi[:, None])

        new_poly_angular_vel = state.polygon.angular_velocity.at[ai].add(a_drv * poly_ai)
        new_poly_angular_vel = new_poly_angular_vel.at[bi].add(b_drv * poly_bi)

        # Circle
        ai -= static_sim_params.num_polygons
        bi -= static_sim_params.num_polygons

        new_circle_vel = state.circle.velocity.at[ai].add(a_dv * jnp.logical_not(poly_ai[:, None]))
        new_circle_vel = new_circle_vel.at[bi].add(b_dv * jnp.logical_not(poly_bi[:, None]))

        new_circle_angular_vel = state.circle.angular_velocity.at[ai].add(a_drv * jnp.logical_not(poly_ai))
        new_circle_angular_vel = new_circle_angular_vel.at[bi].add(b_drv * jnp.logical_not(poly_bi))

        state = state.replace(
            polygon=state.polygon.replace(
                velocity=new_poly_vel,
                angular_velocity=new_poly_angular_vel,
            ),
            circle=state.circle.replace(
                velocity=new_circle_vel,
                angular_velocity=new_circle_angular_vel,
            ),
        )

        return state

    return _resolve_joint_warm_starting


def make_joint_resolver_fn(n_joints, static_sim_params):
    def _resolve_joints(state, params):
        def _calc_single_joint(state, joint_index):
            joint = jax.tree.map(lambda x: x[joint_index], state.joint)

            s1 = select_shape(state, joint.a_index, static_sim_params)
            s2 = select_shape(state, joint.b_index, static_sim_params)

            s1_is_poly = joint.a_index < static_sim_params.num_polygons
            s2_is_poly = joint.b_index < static_sim_params.num_polygons

            (
                a_dv,
                b_dv,
                a_drv,
                b_drv,
                a_dp,
                b_dp,
                jp,
                impulse,
                r_impulse,
            ) = resolve_joint(s1, s2, joint, params)

            state = state.replace(
                joint=state.joint.replace(
                    global_position=state.joint.global_position.at[joint_index].set(jp),
                    acc_impulse=state.joint.acc_impulse.at[joint_index].set(impulse),
                    acc_r_impulse=state.joint.acc_r_impulse.at[joint_index].set(r_impulse),
                ),
            )

            rv = (
                state.polygon.velocity.at[joint.a_index].add(a_dv * s1_is_poly).at[joint.b_index].add(b_dv * s2_is_poly)
            )
            rav = (
                state.polygon.angular_velocity.at[joint.a_index]
                .add(a_drv * s1_is_poly)
                .at[joint.b_index]
                .add(b_drv * s2_is_poly)
            )
            rp = (
                state.polygon.position.at[joint.a_index].add(a_dp * s1_is_poly).at[joint.b_index].add(b_dp * s2_is_poly)
            )

            a_circle_index = joint.a_index - static_sim_params.num_polygons
            b_circle_index = joint.b_index - static_sim_params.num_polygons
            cv = (
                state.circle.velocity.at[a_circle_index]
                .add(a_dv * jnp.logical_not(s1_is_poly))
                .at[b_circle_index]
                .add(b_dv * jnp.logical_not(s2_is_poly))
            )
            cav = (
                state.circle.angular_velocity.at[a_circle_index]
                .add(a_drv * jnp.logical_not(s1_is_poly))
                .at[b_circle_index]
                .add(b_drv * jnp.logical_not(s2_is_poly))
            )
            cp = (
                state.circle.position.at[a_circle_index]
                .add(a_dp * jnp.logical_not(s1_is_poly))
                .at[b_circle_index]
                .add(b_dp * jnp.logical_not(s2_is_poly))
            )

            state = state.replace(
                polygon=state.polygon.replace(
                    velocity=rv,
                    angular_velocity=rav,
                    position=rp,
                ),
                circle=state.circle.replace(
                    velocity=cv,
                    angular_velocity=cav,
                    position=cp,
                ),
            )

            return state, None

        state, _ = jax.lax.scan(_calc_single_joint, state, xs=jnp.arange(n_joints))

        return state

    return _resolve_joints


def calc_nrr(static_sim_params: StaticSimParams):
    nrr_all = static_sim_params.num_polygons * (static_sim_params.num_polygons - 1) // 2
    nrr_sf = static_sim_params.num_static_fixated_polys * (static_sim_params.num_static_fixated_polys - 1) // 2
    nrr = nrr_all - nrr_sf
    return nrr


def get_pairwise_interaction_indices(static_sim_params: StaticSimParams):
    # Pre-compute collision pairs
    # Circle-circle
    ncc = static_sim_params.num_circles * (static_sim_params.num_circles - 1) / 2
    ncc = jnp.round(ncc).astype(int)

    circle_circle_pairs = jnp.zeros((ncc, 2), dtype=jnp.int32)

    ci = 0
    for c1 in range(static_sim_params.num_circles):
        for c2 in range(static_sim_params.num_circles):
            valid = c1 < c2
            # valid = (c1 != c2)
            new_val = valid * jnp.array([c1, c2]) + (1 - valid) * circle_circle_pairs[ci]
            circle_circle_pairs = circle_circle_pairs.at[ci].set(new_val)
            ci += 1 * valid

    # Circle-Polygon
    ncr = static_sim_params.num_circles * static_sim_params.num_polygons

    circle_poly_pairs = jnp.zeros((ncr, 2), dtype=jnp.int32)
    ci = 0
    for c1 in range(static_sim_params.num_circles):
        for r1 in range(static_sim_params.num_polygons):
            circle_poly_pairs = circle_poly_pairs.at[ci].set(jnp.array([c1, r1]))
            ci += 1

    # Polygon-Polygon
    nrr = calc_nrr(static_sim_params)

    poly_poly_pairs = jnp.zeros((nrr, 2), dtype=jnp.int32)

    ci = 0
    for r1 in range(static_sim_params.num_polygons):
        for r2 in range(static_sim_params.num_polygons):
            both_fixated = (
                r1 < static_sim_params.num_static_fixated_polys and r2 < static_sim_params.num_static_fixated_polys
            )
            valid = (r1 < r2) and (not both_fixated)
            new_val = valid * jnp.array([r1, r2]) + (1 - valid) * poly_poly_pairs[ci]
            poly_poly_pairs = poly_poly_pairs.at[ci].set(new_val)
            ci += 1 * valid

    return ncc, ncr, nrr, circle_circle_pairs, circle_poly_pairs, poly_poly_pairs


class PhysicsEngine:
    def __init__(self, static_sim_params: StaticSimParams):
        self.static_sim_params = static_sim_params

        (
            ncc,
            ncr,
            nrr,
            self.circle_circle_pairs,
            self.circle_poly_pairs,
            self.poly_poly_pairs,
        ) = get_pairwise_interaction_indices(static_sim_params)
        batch_size = self.static_sim_params.solver_batch_size

        nrr_b = jnp.ceil(nrr / batch_size).astype(int)
        ncr_b = jnp.ceil(ncr / batch_size).astype(int)
        ncc_b = jnp.ceil(ncc / batch_size).astype(int)

        # Define the impulse resolution functions
        self.rr_fn = make_impulse_resolver_fn(
            static_sim_params,
            True,
            True,
            nrr,
            nrr_b,
            batch_size,
            nrr_b * batch_size,
            self.poly_poly_pairs,
        )
        self.cr_fn = make_impulse_resolver_fn(
            static_sim_params,
            False,
            True,
            ncr,
            ncr_b,
            batch_size,
            ncr_b * batch_size,
            self.circle_poly_pairs,
        )
        self.cc_fn = make_impulse_resolver_fn(
            static_sim_params,
            False,
            False,
            ncc,
            ncc_b,
            batch_size,
            ncc_b * batch_size,
            self.circle_circle_pairs,
        )

        # Define the joint resolver functions
        self.j_fn = make_joint_resolver_fn(self.static_sim_params.num_joints, self.static_sim_params)

        # Define impulse warm starting functions
        self.ws_rr_fn = make_impulse_warm_starting_fn(static_sim_params, True, True, nrr, self.poly_poly_pairs)
        self.ws_cr_fn = make_impulse_warm_starting_fn(static_sim_params, False, True, ncr, self.circle_poly_pairs)
        self.ws_cc_fn = make_impulse_warm_starting_fn(static_sim_params, False, False, ncc, self.circle_circle_pairs)

        # Define joint warm starting functions
        self.ws_j_fn = make_joint_warm_starting_fn(static_sim_params, static_sim_params.num_joints)

    def calculate_collision_manifolds(self, state: SimState):
        def _calc_rr_manifold(rr_indexes, m_index):
            r1 = jax.tree.map(lambda x: x[rr_indexes[0]], state.polygon)
            r2 = jax.tree.map(lambda x: x[rr_indexes[1]], state.polygon)

            warm_start_manifold = jax.tree.map(lambda x: x[m_index], state.acc_rr_manifolds)

            ms = generate_manifolds_polygon_polygon(r1, r2, warm_start_manifold)
            return ms

        rr_manifolds = jax.vmap(_calc_rr_manifold)(self.poly_poly_pairs, jnp.arange(len(self.poly_poly_pairs)))

        def _calc_cr_manifold(cr_indexes, m_index):
            c1 = jax.tree.map(lambda x: x[cr_indexes[0]], state.circle)
            r1 = jax.tree.map(lambda x: x[cr_indexes[1]], state.polygon)

            warm_start_manifold = jax.tree.map(lambda x: x[m_index], state.acc_cr_manifolds)

            m = generate_manifold_circle_polygon(c1, r1, warm_start_manifold)
            return m

        cr_manifolds = jax.vmap(_calc_cr_manifold)(self.circle_poly_pairs, jnp.arange(len(self.circle_poly_pairs)))

        def _calc_cc_manifold(cc_indexes, m_index):
            c1 = jax.tree.map(lambda x: x[cc_indexes[0]], state.circle)
            c2 = jax.tree.map(lambda x: x[cc_indexes[1]], state.circle)

            warm_start_manifold = jax.tree.map(lambda x: x[m_index], state.acc_cc_manifolds)

            m = generate_manifold_circle_circle(c1, c2, warm_start_manifold)
            return m

        cc_manifolds = jax.vmap(_calc_cc_manifold)(self.circle_circle_pairs, jnp.arange(len(self.circle_circle_pairs)))

        return rr_manifolds, cr_manifolds, cc_manifolds

    def step(self, state: SimState, params: SimParams, actions: jnp.ndarray):
        chex.assert_shape(
            actions,
            (self.static_sim_params.num_joints + self.static_sim_params.num_thrusters,),
        )
        motor_actions = actions[: self.static_sim_params.num_joints]
        thruster_actions = actions[self.static_sim_params.num_joints :]

        # Apply gravity
        state = state.replace(
            polygon=state.polygon.replace(
                velocity=state.polygon.velocity + state.gravity * params.dt * (state.polygon.inverse_mass != 0)[:, None]
            ),
            circle=state.circle.replace(
                velocity=state.circle.velocity + state.gravity * params.dt * (state.circle.inverse_mass != 0)[:, None],
            ),
        )

        rr_manifolds, cr_manifolds, cc_manifolds = self.calculate_collision_manifolds(state)

        def _calc_motors(revolute_joint_index):
            revolute_joint = jax.tree.map(lambda x: x[revolute_joint_index], state.joint)

            r1 = select_shape(state, revolute_joint.a_index, self.static_sim_params)
            r2 = select_shape(state, revolute_joint.b_index, self.static_sim_params)

            a_drv, b_drv = apply_motor(r1, r2, revolute_joint, motor_actions[revolute_joint_index], params)
            return (
                revolute_joint.a_index,
                revolute_joint.b_index,
                a_drv,
                b_drv,
                revolute_joint.motor_speed * params.base_motor_speed * revolute_joint.motor_on
                + (1 - revolute_joint.motor_on) * -1,
            )

        ai, bi, a_drv, b_drv, max_speed = jax.vmap(_calc_motors)(jnp.arange(self.static_sim_params.num_joints))

        poly_ai = ai < self.static_sim_params.num_polygons
        poly_bi = bi < self.static_sim_params.num_polygons

        new_poly_angular_vel = state.polygon.angular_velocity.at[ai].add(a_drv * poly_ai)
        new_poly_angular_vel = new_poly_angular_vel.at[bi].add(b_drv * poly_bi)

        ai -= self.static_sim_params.num_polygons
        bi -= self.static_sim_params.num_polygons

        new_circle_angular_vel = state.circle.angular_velocity.at[ai].add(a_drv * jnp.logical_not(poly_ai))
        new_circle_angular_vel = new_circle_angular_vel.at[bi].add(b_drv * jnp.logical_not(poly_bi))

        state = state.replace(
            polygon=state.polygon.replace(
                angular_velocity=new_poly_angular_vel,
            ),
            circle=state.circle.replace(
                angular_velocity=new_circle_angular_vel,
            ),
        )

        # Thrusters
        def calc_thrusters(state):
            def calc_thruster(t_index):
                thruster = jax.tree.map(lambda x: x[t_index], state.thruster)
                parent_shape = select_shape(state, thruster.object_index, self.static_sim_params)

                pos_after_transform = rmat(parent_shape.rotation) @ thruster.relative_position

                rotation = thruster.rotation + parent_shape.rotation
                dipolyion = jnp.array([jnp.cos(rotation), jnp.sin(rotation)])
                force = (
                    thruster.power
                    * thruster.active
                    * params.base_thruster_power
                    * thruster_actions[t_index]
                    * params.dt
                )

                drv = parent_shape.inverse_inertia * jnp.cross(pos_after_transform, dipolyion) * force
                dv = parent_shape.inverse_mass * dipolyion * force
                return thruster.object_index, drv, dv

            ai, drv_i, dv_i = jax.vmap(calc_thruster)(jnp.arange(self.static_sim_params.num_thrusters))
            is_poly = ai < self.static_sim_params.num_polygons

            new_poly_angular_vel = state.polygon.angular_velocity.at[ai].add(drv_i * is_poly)
            new_poly_vel = state.polygon.velocity.at[ai].add(dv_i * is_poly[:, None])
            ai -= self.static_sim_params.num_polygons
            new_circle_angular_vel = state.circle.angular_velocity.at[ai].add(drv_i * jnp.logical_not(is_poly))
            new_circle_vel = state.circle.velocity.at[ai].add(dv_i * jnp.logical_not(is_poly[:, None]))

            state = state.replace(
                polygon=state.polygon.replace(
                    angular_velocity=new_poly_angular_vel,
                    velocity=new_poly_vel,
                ),
                circle=state.circle.replace(
                    angular_velocity=new_circle_angular_vel,
                    velocity=new_circle_vel,
                ),
            )
            return state

        state = calc_thrusters(state)

        # Warm starting
        if self.static_sim_params.do_warm_starting:
            state = self.ws_rr_fn(state, rr_manifolds)
            state = self.ws_cr_fn(state, cr_manifolds)
            state = self.ws_cc_fn(state, cc_manifolds)
            state = self.ws_j_fn(state)

        # Main render loop
        def _solver_iteration(carry, unused):
            state, rr_manifolds, cr_manifolds, cc_manifolds = carry

            # Joints
            state = self.j_fn(state, params)

            # Collisions
            state, rr_manifolds = self.rr_fn(state, rr_manifolds, params)
            state, cr_manifolds = self.cr_fn(state, cr_manifolds, params)
            state, cc_manifolds = self.cc_fn(state, cc_manifolds, params)

            return (state, rr_manifolds, cr_manifolds, cc_manifolds), None

        (state, rr_manifolds, cr_manifolds, cc_manifolds), _ = jax.lax.scan(
            _solver_iteration,
            (state, rr_manifolds, cr_manifolds, cc_manifolds),
            None,
            length=self.static_sim_params.num_solver_iterations,
        )

        if self.static_sim_params.do_warm_starting:
            state = state.replace(
                acc_rr_manifolds=rr_manifolds,
                acc_cr_manifolds=cr_manifolds,
                acc_cc_manifolds=cc_manifolds,
            )

        # Euler step
        state = state.replace(
            polygon=state.polygon.replace(
                position=state.polygon.position + state.polygon.velocity * params.dt,
                rotation=(state.polygon.rotation + state.polygon.angular_velocity * params.dt),  # % (2 * jnp.pi),
            ),
            circle=state.circle.replace(
                position=state.circle.position + state.circle.velocity * params.dt,
                rotation=(state.circle.rotation + state.circle.angular_velocity * params.dt),  # % (2 * jnp.pi),
            ),
        )

        # Clip fast/far objects
        state = clip_state(state, params)

        def _update_thruster_relative_pos(thruster):
            parent_shape = select_shape(state, thruster.object_index, self.static_sim_params)
            return thruster.replace(
                global_position=parent_shape.position
                + jnp.matmul(rmat(parent_shape.rotation), thruster.relative_position)
            )

        state = state.replace(thruster=jax.vmap(_update_thruster_relative_pos)(state.thruster))

        return state, (
            jax.tree.map(lambda x: x[:, 0], rr_manifolds),
            cr_manifolds,
            cc_manifolds,
        )


def get_empty_collision_manifolds(static_sim_params: StaticSimParams):
    nrr = calc_nrr(static_sim_params)
    acc_rr_manifolds = CollisionManifold(
        normal=jnp.zeros((nrr, 2, 2), dtype=jnp.float32),
        penetration=jnp.zeros((nrr, 2), dtype=jnp.float32),
        collision_point=jnp.zeros((nrr, 2, 2), dtype=jnp.float32),
        acc_impulse_normal=jnp.zeros((nrr, 2), dtype=jnp.float32),
        acc_impulse_tangent=jnp.zeros((nrr, 2), dtype=jnp.float32),
        active=jnp.zeros((nrr, 2), dtype=bool),
        restitution_velocity_target=jnp.zeros((nrr, 2), dtype=jnp.float32),
    )

    ncr = static_sim_params.num_polygons * static_sim_params.num_circles
    acc_cr_manifolds = CollisionManifold(
        normal=jnp.zeros((ncr, 2), dtype=jnp.float32),
        penetration=jnp.zeros((ncr), dtype=jnp.float32),
        collision_point=jnp.zeros((ncr, 2), dtype=jnp.float32),
        acc_impulse_normal=jnp.zeros((ncr), dtype=jnp.float32),
        acc_impulse_tangent=jnp.zeros((ncr), dtype=jnp.float32),
        active=jnp.zeros((ncr), dtype=bool),
        restitution_velocity_target=jnp.zeros((ncr,), dtype=jnp.float32),
    )

    ncc = (static_sim_params.num_circles * (static_sim_params.num_circles - 1)) // 2
    acc_cc_manifolds = CollisionManifold(
        normal=jnp.zeros((ncc, 2), dtype=jnp.float32),
        penetration=jnp.zeros((ncc), dtype=jnp.float32),
        collision_point=jnp.zeros((ncc, 2), dtype=jnp.float32),
        acc_impulse_normal=jnp.zeros((ncc), dtype=jnp.float32),
        acc_impulse_tangent=jnp.zeros((ncc), dtype=jnp.float32),
        active=jnp.zeros((ncc), dtype=bool),
        restitution_velocity_target=jnp.zeros((ncc,), dtype=jnp.float32),
    )

    return acc_rr_manifolds, acc_cr_manifolds, acc_cc_manifolds


def create_empty_sim(
    static_sim_params,
    add_floor=True,
    add_walls_and_ceiling=True,
    scene_size=5,
    floor_offset=0.2,
):
    # Polygons
    polygon_pos = jnp.zeros((static_sim_params.num_polygons, 2), dtype=jnp.float32)
    polygon_vertices = jnp.zeros(
        (static_sim_params.num_polygons, static_sim_params.max_polygon_vertices, 2),
        dtype=jnp.float32,
    )
    polygon_vel = jnp.zeros((static_sim_params.num_polygons, 2), dtype=jnp.float32)
    polygon_rotation = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
    polygon_angular_velocity = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
    polygon_inverse_mass = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
    polygon_inverse_inertia = jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32)
    polygon_active = jnp.zeros(static_sim_params.num_polygons, dtype=bool)

    # Circles
    circle_position = jnp.zeros((static_sim_params.num_circles, 2), dtype=jnp.float32)
    circle_radius = jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32)
    circle_vel = jnp.zeros((static_sim_params.num_circles, 2), dtype=jnp.float32)
    circle_inverse_mass = jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32)
    circle_inverse_inertia = jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32)
    circle_active = jnp.zeros(static_sim_params.num_circles, dtype=bool)

    # We simulate half-spaces by just using polygons with large dimensions.
    # Floor
    if add_floor:
        polygon_pos = polygon_pos.at[0].set(jnp.array([scene_size / 2, -scene_size + floor_offset]))
        polygon_vertices = polygon_vertices.at[0].set(
            jnp.array(
                [
                    [scene_size / 2, scene_size + floor_offset],
                    [scene_size / 2, -scene_size - floor_offset],
                    [-scene_size / 2, -scene_size - floor_offset],
                    [-scene_size / 2, scene_size + floor_offset],
                ]
            )
        )

        polygon_inverse_mass = polygon_inverse_mass.at[0].set(0.0)
        polygon_inverse_inertia = polygon_inverse_inertia.at[0].set(0.0)
        polygon_active = polygon_active.at[0].set(True)

    if add_walls_and_ceiling:
        # Side Walls
        polygon_pos = polygon_pos.at[1].set(jnp.array([0.0, 0.0]))
        polygon_vertices = polygon_vertices.at[1].set(
            jnp.array(
                [
                    [-scene_size, scene_size],
                    [0.0, scene_size],
                    [0.0, 0],
                    [-scene_size, 0],
                ]
            )
        )

        polygon_inverse_mass = polygon_inverse_mass.at[1].set(0.0)
        polygon_inverse_inertia = polygon_inverse_inertia.at[1].set(0.0)

        polygon_pos = polygon_pos.at[2].set(jnp.array([0.0, 0.0]))
        polygon_vertices = polygon_vertices.at[2].set(
            jnp.array(
                [
                    [scene_size, scene_size],
                    [2 * scene_size, scene_size],
                    [2 * scene_size, 0],
                    [scene_size, 0],
                ]
            )
        )

        polygon_inverse_mass = polygon_inverse_mass.at[2].set(0.0)
        polygon_inverse_inertia = polygon_inverse_inertia.at[2].set(0.0)

        # Ceiling
        polygon_pos = polygon_pos.at[3].set(jnp.array([scene_size / 2, floor_offset + 2 * scene_size]))
        polygon_vertices = polygon_vertices.at[3].set(
            jnp.array(
                [
                    [scene_size, scene_size + floor_offset],
                    [scene_size / 2, -scene_size - floor_offset],
                    [-(scene_size / 2), -scene_size - floor_offset],
                    [-(scene_size / 2), scene_size + floor_offset],
                ]
            )
        )

        polygon_inverse_mass = polygon_inverse_mass.at[3].set(0.0)
        polygon_inverse_inertia = polygon_inverse_inertia.at[3].set(0.0)

        polygon_active = polygon_active.at[1:4].set(True)

    # Joints
    revolute_joint_a_pos = jnp.zeros((static_sim_params.num_joints, 2), dtype=jnp.float32)
    revolute_joint_b_pos = jnp.zeros((static_sim_params.num_joints, 2), dtype=jnp.float32)
    revolute_joint_a_index = jnp.zeros(static_sim_params.num_joints, dtype=jnp.int32)
    revolute_joint_b_index = jnp.zeros(static_sim_params.num_joints, dtype=jnp.int32)

    joints = Joint(
        a_index=revolute_joint_a_index,
        b_index=revolute_joint_b_index,
        a_relative_pos=revolute_joint_a_pos,
        b_relative_pos=revolute_joint_b_pos,
        active=jnp.zeros(static_sim_params.num_joints, dtype=bool),
        global_position=jnp.ones((static_sim_params.num_joints, 2), dtype=jnp.float32) * 256,
        motor_on=jnp.zeros(static_sim_params.num_joints, dtype=bool),
        motor_speed=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
        motor_power=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
        acc_impulse=jnp.zeros((static_sim_params.num_joints, 2), dtype=jnp.float32),
        motor_has_joint_limits=jnp.zeros(static_sim_params.num_joints, dtype=bool),
        min_rotation=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
        max_rotation=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
        is_fixed_joint=jnp.zeros(static_sim_params.num_joints, dtype=bool),
        rotation=jnp.zeros(static_sim_params.num_joints, dtype=jnp.float32),
        acc_r_impulse=jnp.zeros((static_sim_params.num_joints), dtype=jnp.float32),
    )

    thrusters = Thruster(
        object_index=jnp.zeros(static_sim_params.num_thrusters, dtype=jnp.int32),
        relative_position=jnp.zeros((static_sim_params.num_thrusters, 2), dtype=jnp.float32),
        active=jnp.zeros(static_sim_params.num_thrusters, dtype=bool),
        power=jnp.zeros(static_sim_params.num_thrusters, dtype=jnp.float32),
        global_position=jnp.zeros((static_sim_params.num_thrusters, 2), dtype=jnp.float32),
        rotation=jnp.zeros(static_sim_params.num_thrusters, dtype=jnp.float32),
    )

    collision_matrix = calculate_collision_matrix(static_sim_params, joints)

    (
        acc_rr_manifolds,
        acc_cr_manifolds,
        acc_cc_manifolds,
    ) = get_empty_collision_manifolds(static_sim_params)

    n_vertices = jnp.ones((static_sim_params.num_polygons,), dtype=jnp.int32) * 4
    state = SimState(
        polygon=RigidBody(
            position=polygon_pos,
            vertices=polygon_vertices,
            n_vertices=n_vertices,
            velocity=polygon_vel * 0,
            inverse_mass=polygon_inverse_mass,
            rotation=polygon_rotation,
            angular_velocity=polygon_angular_velocity,
            inverse_inertia=polygon_inverse_inertia,
            friction=jnp.ones(static_sim_params.num_polygons),
            restitution=jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32),
            radius=jnp.zeros(static_sim_params.num_polygons, dtype=jnp.float32),
            active=polygon_active,
            collision_mode=jnp.ones(static_sim_params.num_polygons, dtype=int)
            .at[: static_sim_params.num_static_fixated_polys]
            .set(2),
        ),
        circle=RigidBody(
            radius=circle_radius,
            position=circle_position,
            velocity=circle_vel,
            inverse_mass=circle_inverse_mass,
            inverse_inertia=circle_inverse_inertia,
            rotation=jnp.ones(static_sim_params.num_circles) * 0,
            angular_velocity=jnp.ones(static_sim_params.num_circles) * 0,
            friction=jnp.ones(static_sim_params.num_circles),
            restitution=jnp.zeros(static_sim_params.num_circles, dtype=jnp.float32),
            vertices=jnp.zeros(
                (
                    static_sim_params.num_circles,
                    static_sim_params.max_polygon_vertices,
                    2,
                ),
                dtype=jnp.float32,
            ),
            n_vertices=jnp.zeros((static_sim_params.num_circles,), dtype=jnp.int32),
            active=circle_active,
            collision_mode=jnp.ones(static_sim_params.num_circles, dtype=int),
        ),
        thruster=thrusters,
        joint=joints,
        collision_matrix=collision_matrix,
        acc_rr_manifolds=acc_rr_manifolds,
        acc_cr_manifolds=acc_cr_manifolds,
        acc_cc_manifolds=acc_cc_manifolds,
        gravity=jnp.array([0.0, -9.81]),
    )

    polygon_densities = jnp.ones(static_sim_params.num_polygons)
    circle_densities = jnp.ones(static_sim_params.num_circles)

    state = recalculate_mass_and_inertia(state, static_sim_params, polygon_densities, circle_densities)

    return state
