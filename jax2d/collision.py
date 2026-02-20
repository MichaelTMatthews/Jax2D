import math

import jax
import jax.numpy as jnp

from jax2d.sim_state import RigidBody, CollisionManifold, SimParams
from jax2d.maths import vs_cross, sv_cross, rmat, zero_to_one


def should_collide(a: RigidBody, b: RigidBody):
    return ((a.collision_mode == 2) | (b.collision_mode == 2)) | (a.collision_mode * b.collision_mode > 0)


def resolve_warm_starting_impulse(
    a: RigidBody,
    b: RigidBody,
    m: CollisionManifold,
    does_collide: bool,
):
    r1 = m.collision_point - a.position
    r2 = m.collision_point - b.position

    tangent = vs_cross(m.normal, 1.0)
    impulse = m.acc_impulse_normal * m.normal + m.acc_impulse_tangent * tangent
    a_dv = -impulse * a.inverse_mass
    b_dv = impulse * b.inverse_mass
    a_drv = -a.inverse_inertia * jnp.cross(r1, impulse)
    b_drv = b.inverse_inertia * jnp.cross(r2, impulse)

    should_resolve = (
        m.active & jnp.logical_not((a.inverse_mass == 0) & (b.inverse_mass == 0)) & does_collide & a.active & b.active
    )

    rvals = (a_dv, b_dv, a_drv, b_drv)

    return jax.tree.map(lambda x: jax.lax.select(should_resolve, x, jnp.zeros_like(x)), rvals)


def resolve_collision(
    a: RigidBody,
    b: RigidBody,
    collision_manifold: CollisionManifold,
    does_collide: bool,
    sim_params: SimParams,
):
    # Calculate useful things
    r1 = collision_manifold.collision_point - a.position
    r2 = collision_manifold.collision_point - b.position

    av = a.velocity + sv_cross(a.angular_velocity, r1)
    bv = b.velocity + sv_cross(b.angular_velocity, r2)

    dv = bv - av
    vn = jnp.dot(dv, collision_manifold.normal)

    rn1 = jnp.dot(r1, collision_manifold.normal)
    rn2 = jnp.dot(r2, collision_manifold.normal)

    inv_mass_normal = (
        a.inverse_mass
        + b.inverse_mass
        + a.inverse_inertia * (jnp.dot(r1, r1) - rn1 * rn1)
        + b.inverse_inertia * (jnp.dot(r2, r2) - rn2 * rn2)
    )

    # Calculate the target bias velocity as a function of penetration
    bias = (
        -sim_params.baumgarte_coefficient_collision
        / sim_params.dt
        * jnp.minimum(0.0, -collision_manifold.penetration + sim_params.slop)
    )

    # Calculate the impulse along the collision normal by clamping the accumulated impulse
    impulse_normal_mag = (-vn + bias - collision_manifold.restitution_velocity_target) / zero_to_one(inv_mass_normal)
    new_acc_impulse_normal = jnp.maximum(collision_manifold.acc_impulse_normal + impulse_normal_mag, 0.0)
    impulse_normal_mag = new_acc_impulse_normal - collision_manifold.acc_impulse_normal
    impulse_normal = impulse_normal_mag * collision_manifold.normal

    # Apply the impulse along the normal
    a_dv = -a.inverse_mass * impulse_normal
    b_dv = b.inverse_mass * impulse_normal

    a_drv = -a.inverse_inertia * jnp.cross(r1, impulse_normal)
    b_drv = b.inverse_inertia * jnp.cross(r2, impulse_normal)

    # Recalculate dv with the applied normal impulse taken into account
    dv = (
        (b.velocity + b_dv)
        + sv_cross(b.angular_velocity + b_drv, r2)
        - (a.velocity + a_dv)
        - sv_cross(a.angular_velocity + a_drv, r1)
    )

    # Calculate speed along the tangent in order to calculate friction
    tangent = vs_cross(collision_manifold.normal, 1.0)
    vt = jnp.dot(dv, tangent)

    rt1 = jnp.dot(r1, tangent)
    rt2 = jnp.dot(r2, tangent)
    inv_mass_tangent = (
        a.inverse_mass
        + b.inverse_mass
        + a.inverse_inertia * (jnp.dot(r1, r1) - rt1 * rt1)
        + b.inverse_inertia * (jnp.dot(r2, r2) - rt2 * rt2)
    )

    # Calculate friction coefficient between the two shapes
    mu = jnp.sqrt(jnp.square(a.friction * sim_params.base_friction) + jnp.square(b.friction * sim_params.base_friction))

    # Calculate friction impulse along the tangent using Coulomb's Law applied to the accumulated impulse
    impulse_tangent_mag = -vt / zero_to_one(inv_mass_tangent)
    max_friction_impulse = mu * new_acc_impulse_normal
    new_acc_impulse_tangent = jnp.clip(
        collision_manifold.acc_impulse_tangent + impulse_tangent_mag,
        -max_friction_impulse,
        max_friction_impulse,
    )
    impulse_tangent_mag = new_acc_impulse_tangent - collision_manifold.acc_impulse_tangent
    impulse_tangent = impulse_tangent_mag * tangent

    # Apply the impulse along the tangent (friction impulse)
    a_dv -= a.inverse_mass * impulse_tangent
    a_drv -= a.inverse_inertia * jnp.cross(r1, impulse_tangent)

    b_dv += b.inverse_mass * impulse_tangent
    b_drv += b.inverse_inertia * jnp.cross(r2, impulse_tangent)

    # Return zeros if not colliding
    rvals = (a_dv, b_dv, a_drv, b_drv, new_acc_impulse_normal, new_acc_impulse_tangent)

    should_resolve = (
        collision_manifold.active
        & jnp.logical_not((a.inverse_mass == 0) & (b.inverse_mass == 0))
        & does_collide
        & a.active
        & b.active
    )

    return jax.tree.map(lambda x: jax.lax.select(should_resolve, x, jnp.zeros_like(x)), rvals)


def _calc_relative_velocity(v1, com1, av1, v2, com2, av2, cpoint, normal):
    av = v1 + sv_cross(av1, cpoint - com1)
    bv = v2 + sv_cross(av2, cpoint - com2)

    vn = jnp.dot(bv - av, normal)

    return vn


def generate_manifold_circle_circle(a: RigidBody, b: RigidBody, ws_manifold: CollisionManifold):
    n = b.position - a.position
    dist = jnp.linalg.norm(n)
    r = a.radius + b.radius

    is_colliding = (dist < r) & a.active & b.active & should_collide(a, b)

    penetration = r - dist
    normal = n / zero_to_one(dist)  # Arbitrary value if two circles have the same position

    collision_point = a.position + normal * a.radius

    vn = _calc_relative_velocity(
        a.velocity,
        a.position,
        a.angular_velocity,
        b.velocity,
        b.position,
        b.angular_velocity,
        collision_point,
        normal,
    )
    v_rest = vn * jnp.minimum(a.restitution, b.restitution)

    cm = CollisionManifold(
        normal=normal,
        penetration=penetration,
        active=is_colliding,
        collision_point=collision_point,
        acc_impulse_normal=jax.lax.select(
            ws_manifold.active & is_colliding,
            ws_manifold.acc_impulse_normal,
            jnp.zeros_like(ws_manifold.acc_impulse_normal),
        ),
        acc_impulse_tangent=jax.lax.select(
            ws_manifold.active & is_colliding,
            ws_manifold.acc_impulse_tangent,
            jnp.zeros_like(ws_manifold.acc_impulse_tangent),
        ),
        restitution_velocity_target=v_rest,
    )

    return jax.tree.map(lambda x: jax.lax.select(is_colliding, x, jnp.zeros_like(x)), cm)


def generate_manifold_circle_polygon(circle: RigidBody, polygon: RigidBody, ws_manifold: CollisionManifold):
    # Transform the circle into the local space of the polygon
    poly_M = rmat(polygon.rotation)
    circle_centre = jnp.matmul(poly_M.transpose((1, 0)), circle.position - polygon.position)

    def _clip_point_to_line(point, line_a, line_b):
        def _signed_line_distance(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        dist = _signed_line_distance(point, line_a, line_b)

        along_line = line_b - line_a
        norm = jnp.linalg.norm(along_line)
        along_line /= zero_to_one(norm)

        dot = jnp.dot(along_line, point - line_a)
        dot_clipped = jnp.clip(dot, 0.0, jnp.linalg.norm(line_b - line_a))

        clipped_point = line_a + dot_clipped * along_line

        return dist < 0, clipped_point, jnp.linalg.norm(clipped_point - point)

    # We clip the centre of the circle to the edges of the polygon
    # This assumes the centre is outside the polygon
    next_vertices = jnp.concatenate([polygon.vertices[1:], polygon.vertices[:1]], axis=0)
    next_vertices = next_vertices.at[polygon.n_vertices - 1].set(polygon.vertices[0])

    in_a, clip_a, dist_a = _clip_point_to_line(circle_centre, polygon.vertices[0], next_vertices[0])
    in_b, clip_b, dist_b = _clip_point_to_line(circle_centre, polygon.vertices[1], next_vertices[1])
    in_c, clip_c, dist_c = _clip_point_to_line(circle_centre, polygon.vertices[2], next_vertices[2])
    in_d, clip_d, dist_d = _clip_point_to_line(circle_centre, polygon.vertices[3], next_vertices[3])

    # If the centre is inside the polygon, then we clip it to the single closest edge
    clips = jnp.array([clip_a, clip_b, clip_c, clip_d])

    dists = jnp.array([dist_a, dist_b, dist_c, dist_d])
    dists = dists.at[3].set(jax.lax.select(polygon.n_vertices == 3, jnp.max(dists[:3]) + 1, dist_d))
    closest_edge = jnp.argmin(dists)
    inside_clipped_outward_point = clips[closest_edge]

    inside = in_a & in_b & in_c & (in_d | (polygon.n_vertices == 3))
    closest = inside_clipped_outward_point

    normal = circle_centre - closest
    d = jnp.linalg.norm(normal)
    r = circle.radius

    active = ((d <= r) | inside) & circle.active & polygon.active & should_collide(polygon, circle)

    # Transform normal back to world space and reverse if circle inside polygon
    normal = jnp.matmul(poly_M, normal) * jax.lax.select(inside, -1, 1)
    norm_of_normal = jnp.linalg.norm(normal)
    normal = -normal / zero_to_one(norm_of_normal)

    collision_point = jnp.matmul(poly_M, closest) + polygon.position

    vn = _calc_relative_velocity(
        circle.velocity,
        circle.position,
        circle.angular_velocity,
        polygon.velocity,
        polygon.position,
        polygon.angular_velocity,
        collision_point,
        normal,
    )
    v_rest = vn * jnp.minimum(circle.restitution, polygon.restitution)

    cm = CollisionManifold(
        normal=normal,
        penetration=jax.lax.select(inside, r, r - d),
        collision_point=collision_point,
        active=active,
        acc_impulse_normal=jax.lax.select(
            ws_manifold.active & active,
            ws_manifold.acc_impulse_normal,
            jnp.zeros_like(ws_manifold.acc_impulse_normal),
        ),
        acc_impulse_tangent=jax.lax.select(
            ws_manifold.active & active,
            ws_manifold.acc_impulse_tangent,
            jnp.zeros_like(ws_manifold.acc_impulse_tangent),
        ),
        restitution_velocity_target=v_rest,
    )
    return jax.tree.map(lambda x: jax.lax.select(active, x, jnp.zeros_like(x)), cm)


def generate_manifolds_polygon_polygon(a: RigidBody, b: RigidBody, ws_manifolds: CollisionManifold):
    # Find axes of least penetration
    a_sep, a_face_index, a_incident_face = find_axis_of_least_penetration(a, b)
    b_sep, b_face_index, b_incident_face = find_axis_of_least_penetration(b, a)

    epsilon = 0.01  # Arbitrary bias to stop collision point flip flopping around

    a_has_most_pen = a_sep + epsilon < b_sep
    most_sep = jnp.maximum(a_sep, b_sep)
    is_colliding = (most_sep < 0) & a.active & b.active & should_collide(a, b)

    # Calculate reference and incident faces
    a_M = rmat(a.rotation)
    b_M = rmat(b.rotation)

    # a_half_dim = jnp.array([0.5, 0.5])
    # b_half_dim = jnp.array([0.5, 0.5])

    b_v_world_space = (
        jnp.array(
            [
                jnp.matmul(b_M, b.vertices[0]),
                jnp.matmul(b_M, b.vertices[1]),
                jnp.matmul(b_M, b.vertices[2]),
                jnp.matmul(b_M, b.vertices[3]),
            ]
        )
        + b.position
    )

    a_v_world_space = (
        jnp.array(
            [
                jnp.matmul(a_M, a.vertices[0]),
                jnp.matmul(a_M, a.vertices[1]),
                jnp.matmul(a_M, a.vertices[2]),
                jnp.matmul(a_M, a.vertices[3]),
            ]
        )
        + a.position
    )

    a_face = jnp.array(
        [
            a_v_world_space[a_face_index],
            a_v_world_space[(a_face_index + 1) % a.n_vertices],
        ]
    )
    b_face = jnp.array(
        [
            b_v_world_space[b_face_index],
            b_v_world_space[(b_face_index + 1) % b.n_vertices],
        ]
    )

    ref_face = jax.lax.select(a_has_most_pen, b_face, a_face)
    # Incident face is composed of the two vertices that have the least penetration into the reference face
    # We can see that this will always be a valid rectangle face and not a diagonal by **thinking hard**
    # ^This is true for rectangles, I *think* it's also true for convex polygons (??)
    incident_face = jax.lax.select(a_has_most_pen, b_incident_face, a_incident_face)

    # Translate ref and incident face to reference space
    r1_angle = math.pi + jnp.arctan2((ref_face[0] - ref_face[1])[0], (ref_face[0] - ref_face[1])[1])
    r1_M = rmat(r1_angle)
    r1_r2_len = jnp.linalg.norm(ref_face[0] - ref_face[1])

    # R0 is (0, 0)
    # R1 is (0, r1_r2_len)

    incident_face_ref_space = jnp.array(
        [
            jnp.matmul(r1_M, incident_face[0] - ref_face[0]),
            jnp.matmul(r1_M, incident_face[1] - ref_face[0]),
        ]
    )

    # Clip incident face to reference face boundaries
    clipped_incident_face_ref_space = jnp.array(
        [
            jnp.clip(
                incident_face_ref_space[0],
                min=jnp.array([-99999, 0]),
                max=jnp.array([999999, r1_r2_len]),
            ),
            jnp.clip(
                incident_face_ref_space[1],
                min=jnp.array([-99999, 0]),
                max=jnp.array([999999, r1_r2_len]),
            ),
        ]
    )

    collision_point_index = jnp.argmax(clipped_incident_face_ref_space[:, 0])
    both_points_in_neg_space = jnp.min(clipped_incident_face_ref_space[:, 0]) > 0

    collision_point1_ref_space = clipped_incident_face_ref_space[collision_point_index]
    collision_point2_ref_space = clipped_incident_face_ref_space[1 - collision_point_index]

    collision_point1 = jnp.matmul(r1_M.transpose((1, 0)), collision_point1_ref_space) + ref_face[0]
    collision_point2 = jnp.matmul(r1_M.transpose((1, 0)), collision_point2_ref_space) + ref_face[0]

    rot_left = rmat(math.pi / 2.0)

    next_a_vertices = jnp.concatenate([a.vertices[1:], a.vertices[:1]])
    next_a_vertices = next_a_vertices.at[a.n_vertices - 1].set(a.vertices[0])

    next_b_vertices = jnp.concatenate([b.vertices[1:], b.vertices[:1]])
    next_b_vertices = next_b_vertices.at[b.n_vertices - 1].set(b.vertices[0])

    a_normals = _calc_normals(rot_left, next_a_vertices, a.vertices)

    b_normals = _calc_normals(rot_left, next_b_vertices, b.vertices)
    norm = jax.lax.select(
        a_has_most_pen,
        -jnp.matmul(b_M, b_normals[b_face_index]),
        jnp.matmul(a_M, a_normals[a_face_index]),
    )

    ws_cm1 = jax.tree.map(lambda x: x[0], ws_manifolds)
    ws_cm2 = jax.tree.map(lambda x: x[1], ws_manifolds)

    vn1 = _calc_relative_velocity(
        a.velocity,
        a.position,
        a.angular_velocity,
        b.velocity,
        b.position,
        b.angular_velocity,
        collision_point1,
        norm,
    )
    v_rest1 = vn1 * jnp.minimum(a.restitution, b.restitution)

    vn2 = _calc_relative_velocity(
        a.velocity,
        a.position,
        a.angular_velocity,
        b.velocity,
        b.position,
        b.angular_velocity,
        collision_point2,
        norm,
    )
    v_rest2 = vn2 * jnp.minimum(a.restitution, b.restitution)

    cm1 = CollisionManifold(
        normal=norm,
        penetration=-most_sep,
        collision_point=collision_point1,
        acc_impulse_normal=jax.lax.select(
            ws_cm1.active & is_colliding,
            ws_cm1.acc_impulse_normal,
            jnp.zeros_like(ws_cm1.acc_impulse_normal),
        ),
        acc_impulse_tangent=jax.lax.select(
            ws_cm1.active & is_colliding,
            ws_cm1.acc_impulse_tangent,
            jnp.zeros_like(ws_cm1.acc_impulse_tangent),
        ),
        active=is_colliding,
        restitution_velocity_target=v_rest1,
    )

    cm2 = CollisionManifold(
        normal=norm,
        penetration=-most_sep,
        collision_point=collision_point2,
        acc_impulse_normal=jax.lax.select(
            ws_cm2.active & (is_colliding & both_points_in_neg_space),
            ws_cm2.acc_impulse_normal,
            jnp.zeros_like(ws_cm2.acc_impulse_normal),
        ),
        acc_impulse_tangent=jax.lax.select(
            ws_cm2.active & (is_colliding & both_points_in_neg_space),
            ws_cm2.acc_impulse_tangent,
            jnp.zeros_like(ws_cm2.acc_impulse_tangent),
        ),
        active=is_colliding & both_points_in_neg_space,
        restitution_velocity_target=v_rest2,
    )

    cm1 = jax.tree.map(lambda x: jax.lax.select(is_colliding, x, jnp.zeros_like(x)), cm1)
    cm2 = jax.tree.map(
        lambda x: jax.lax.select(is_colliding & both_points_in_neg_space, x, jnp.zeros_like(x)),
        cm2,
    )
    return jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y], axis=0), cm1, cm2)


def _calc_normals(matrix, next_vert, vert):
    delta = next_vert - vert
    norm = jax.vmap(jnp.linalg.norm)(delta)

    def _matmul(mat, v, norm):
        return jnp.matmul(mat, v) / zero_to_one(norm)

    return jax.vmap(_matmul, (None, 0, 0))(matrix, delta, norm)


def find_axis_of_least_penetration(a: RigidBody, b: RigidBody):
    # Transform B in A's model space
    # We can then use A without rotation as area and basis vectors as normals

    b_M = rmat(b.rotation)
    a_M = rmat(a.rotation)

    b_v_world_space = (
        jnp.array(
            [
                jnp.matmul(b_M, b.vertices[0]),
                jnp.matmul(b_M, b.vertices[1]),
                jnp.matmul(b_M, b.vertices[2]),
                jnp.matmul(b_M, b.vertices[3]),
            ]
        )
        + b.position
    )

    b_v_a_space = jnp.array(
        [
            jnp.matmul(a_M.transpose((1, 0)), b_v_world_space[0] - a.position),
            jnp.matmul(a_M.transpose((1, 0)), b_v_world_space[1] - a.position),
            jnp.matmul(a_M.transpose((1, 0)), b_v_world_space[2] - a.position),
            jnp.matmul(a_M.transpose((1, 0)), b_v_world_space[3] - a.position),
        ]
    )

    a_v_a_space = a.vertices

    rot_left = rmat(math.pi / 2.0)
    next_a_vertices = jnp.concatenate([a.vertices[1:], a.vertices[:1]])
    next_a_vertices = next_a_vertices.at[a.n_vertices - 1].set(a.vertices[0])

    next_b_vertices = jnp.concatenate([b.vertices[1:], b.vertices[:1]])
    next_b_vertices = next_b_vertices.at[b.n_vertices - 1].set(b.vertices[0])

    a_normals = _calc_normals(rot_left, next_a_vertices, a.vertices)

    highest_separation = -99999
    highest_sep_axis_index = -1
    highest_sep_incident_face_indexes = jnp.array([0, 0], dtype=jnp.int32)

    for i, (normal_a) in enumerate(a_normals):
        a_v_on_axis = jnp.array(
            [
                jnp.dot(normal_a, a_v_a_space[0]),
                jnp.dot(normal_a, a_v_a_space[1]),
                jnp.dot(normal_a, a_v_a_space[2]),
                jnp.dot(normal_a, a_v_a_space[3]),
            ]
        )

        b_v_on_axis = jnp.array(
            [
                jnp.dot(normal_a, b_v_a_space[0]),
                jnp.dot(normal_a, b_v_a_space[1]),
                jnp.dot(normal_a, b_v_a_space[2]),
                jnp.dot(normal_a, b_v_a_space[3]),
            ]
        )

        a_v_on_axis = a_v_on_axis.at[3].set(
            jax.lax.select(
                a.n_vertices == 3,
                a_v_on_axis[:3].min() - 1,
                a_v_on_axis[3],  # -1 for epsilon to ensure that the min/max is not affected
            )
        )
        b_v_on_axis = b_v_on_axis.at[3].set(
            jax.lax.select(
                b.n_vertices == 3,
                b_v_on_axis[:3].max() + 1,
                b_v_on_axis[3],  # +1 for epsilon to ensure that the min/max is not affected
            )
        )

        separation = b_v_on_axis.min() - a_v_on_axis.max()

        incident_face_indexes = b_v_on_axis.argsort()[:2]

        is_highest = separation > highest_separation
        is_highest &= i < a.n_vertices

        highest_separation = is_highest * separation + (1 - is_highest) * highest_separation
        highest_sep_axis_index = is_highest * i + (1 - is_highest) * highest_sep_axis_index
        highest_sep_incident_face_indexes = (
            is_highest * incident_face_indexes + (1 - is_highest) * highest_sep_incident_face_indexes
        )

    highest_sep_incident_face = jnp.array(
        [
            b_v_world_space[highest_sep_incident_face_indexes[0]],
            b_v_world_space[highest_sep_incident_face_indexes[1]],
        ]
    )

    return highest_separation, highest_sep_axis_index, highest_sep_incident_face
