"""State types for jax_evogym.

All state represented as NamedTuples (automatic JAX pytree registration).
SimState = single-robot state (vmap adds batch dim).
SpringTopology + PhysicsConstants + ActuatorInfo = shared across batch.
"""

from __future__ import annotations

from typing import Literal, NamedTuple
import jax.numpy as jnp
import numpy as np


class PhysicsConstants(NamedTuple):
    """Physics parameters. Shared across batch (not vmapped)."""
    dt: float
    gravity: float
    viscous_drag: float
    collision_const_ground: float
    collision_const_obj: float
    collision_vel_damping: float
    ground_vel_damping: float            # ground normal velocity damping (0 in C++)
    collision_base_dist: float
    dynamic_friction_const: float
    friction_const: float            # obj-obj friction (1200)
    ground_friction_const: float     # ground friction (10, hardcoded in C++)
    actuator_convergence: float      # spring rest-length convergence factor
    integrator: int                  # 0=rk4, 1=symplectic_euler
    static_friction_const: float = 0.0
    terrain_stiction_stiffness: float = 0.0
    ground_stiction_stiffness: float = 0.0
    terrain_stiction_switch_speed: float = 0.0
    ground_stiction_switch_speed: float = 0.0
    floor_static_friction_const: float = 0.0
    floor_stiction_stiffness: float = 0.0
    floor_stiction_switch_speed: float = 0.0
    ground_static_friction_const: float = 0.0
    terrain_stiction_scope: int = 0  # slopes are geometry-only; slope-only is currently inert
    friction_model: int = 0          # 0=stateless, 1=projected anchor, 2=projected deformation
    anchor_stiffness: float = 0.0
    deformation_stiffness: float = 0.0
    deformation_damping: float = 0.0
    deformation_decay: float = 1.0
    deformation_normal_bleed: float = 1.0
    breakaway_threshold: float = 0.0
    stiction_enabled: bool = False     # master switch for terrain stiction and memory/anchor friction
    friction_constraint_mode: int = 0  # 0=off, 1=terrain-pbd-anchor
    friction_anchor_stiffness: float = 0.0
    friction_anchor_correction: float = 1.0
    friction_anchor_persist_substeps: int = 10
    slope_contact_mode: int = 0      # retained compatibility no-op
    slope_contact_near_dist: float = 0.01
    slope_contact_release_speed: float = 0.05
    slope_contact_normal_damping: float = 10.0
    slope_contact_tangent_damping: float = 10.0
    slope_contact_assist_mu: float = 0.5
    slope_grip_enabled: bool = False # retained compatibility no-op
    slope_grip_static_friction_const: float = 0.5
    slope_grip_support_stiffness: float = 20_000.0
    slope_grip_switch_speed: float = 0.12
    slope_grip_near_contact_dist: float = 0.08
    slope_grip_near_release_speed: float = 0.25
    slope_grip_scope: int = 0
    diagnostics_enabled: bool = True


class SpringTopology(NamedTuple):
    """Spring connectivity. Shared across batch (not vmapped).
    Indices into the point arrays of SimState."""
    a_idx: jnp.ndarray   # (n_springs,) int32 — first endpoint
    b_idx: jnp.ndarray   # (n_springs,) int32 — second endpoint


class ActuatorInfo(NamedTuple):
    """Per-cell actuator control matching C++ PhysicsEngine.cpp:38-97.

    One action per actuator cell. Shared edges get averaged actions.
    Shared across batch (not vmapped).
    """
    cell_spring_indices: jnp.ndarray    # (n_actuator_cells, 2) int32
    spring_act_count: jnp.ndarray       # (n_springs,) float32 — 0, 1, or 2
    actuated_spring_mask: jnp.ndarray   # (n_springs,) bool


class PerAxisActuatorInfo(NamedTuple):
    """Per-cell actuator control with independent horizontal/vertical actions."""

    h_spring_pairs: jnp.ndarray         # (H*W, 2) int32 — [h_bottom, h_top]
    v_spring_pairs: jnp.ndarray         # (H*W, 2) int32 — [v_left, v_right]
    h_compact_cell_indices: jnp.ndarray  # (N_compact_max,) int32 — world-flat cell idx
    v_compact_cell_indices: jnp.ndarray  # (N_compact_max,) int32 — world-flat cell idx
    h_compact_spring_pairs: jnp.ndarray  # (N_compact_max, 2) int32
    v_compact_spring_pairs: jnp.ndarray  # (N_compact_max, 2) int32
    h_compact_count: jnp.ndarray        # () int32
    v_compact_count: jnp.ndarray        # () int32
    actuated_spring_mask: jnp.ndarray   # (n_springs,) bool
    spring_act_count: jnp.ndarray       # (n_springs,) float32


class CollisionData(NamedTuple):
    """Precomputed collision geometry for object-object collision detection.

    Boxel = one voxel's quad. Corners ordered [BL, BR, TR, TL].
    Edges ordered [bottom, right, top, left].
    Edge e: from corner (e+1)%4 to corner e — outward normals via (-slope.y, slope.x).
    """
    # Boxel geometry
    boxel_corners: jnp.ndarray        # (n_boxels, 4) int32 — point indices
    boxel_edge_a: jnp.ndarray         # (n_boxels, 4) int32 — edge start point indices
    boxel_edge_b: jnp.ndarray         # (n_boxels, 4) int32 — edge end point indices

    # Masks
    boxel_mask: jnp.ndarray           # (n_boxels,) bool — active boxels
    surface_edge_mask: jnp.ndarray    # (n_boxels, 4) bool — per-object surface edges
    boxel_object_id: jnp.ndarray      # (n_boxels,) int32 — object ownership
    boxel_is_robot: jnp.ndarray       # (n_boxels,) bool — robot ownership
    boxel_world_vy: jnp.ndarray       # (n_boxels,) int32 — world grid y
    boxel_world_vx: jnp.ndarray       # (n_boxels,) int32 — world grid x

    # Corner-to-edges mapping: corner k connects edges corner_edge_indices[k]
    corner_edge_indices: jnp.ndarray  # (4, 2) int32

    # Real point count (before expansion for virtual terrain indices)
    n_real_points: int

    # --- Indexed collision triples (fixed-size buffers) ---
    triple_i: jnp.ndarray             # (MAX_TRIPLES,) int32 — main boxel
    triple_j: jnp.ndarray             # (MAX_TRIPLES,) int32 — ref boxel
    triple_k: jnp.ndarray             # (MAX_TRIPLES,) int32 — corner of main (0..3)
    triple_active: jnp.ndarray        # (MAX_TRIPLES,) bool
    triple_pair_slot: jnp.ndarray     # (MAX_TRIPLES,) int32 — dynamic candidate pair slot

    self_triple_i: jnp.ndarray        # (MAX_SELF_TRIPLES,) int32
    self_triple_j: jnp.ndarray        # (MAX_SELF_TRIPLES,) int32
    self_triple_k: jnp.ndarray        # (MAX_SELF_TRIPLES,) int32
    self_triple_active: jnp.ndarray   # (MAX_SELF_TRIPLES,) bool

    # Dynamic broadphase sparse candidate pair list (derived from triple pairs)
    candidate_pair_i: jnp.ndarray     # (MAX_TRIPLES,) int32
    candidate_pair_j: jnp.ndarray     # (MAX_TRIPLES,) int32
    candidate_pair_active: jnp.ndarray  # (MAX_TRIPLES,) bool

    n_triples: jnp.ndarray | int      # actual count (diagnostic only)
    n_self_triples: jnp.ndarray | int # actual count (diagnostic only)
    n_candidate_pairs: jnp.ndarray | int  # actual dynamic candidate pair count
    n_robot_surface_boxels: jnp.ndarray | int  # robot surface boxel count


class StaticColliderData(NamedTuple):
    """Static polygon collider geometry for fixed terrain omitted from SimState."""

    point_positions: jnp.ndarray      # (n_static_points, 2) float32
    cell_vertices: jnp.ndarray        # (n_static_cells, 4) int32 — local to point_positions
    cell_vertex_count: jnp.ndarray    # (n_static_cells,) int32 — 3 or 4
    cell_edge_a: jnp.ndarray          # (n_static_cells, 4) int32
    cell_edge_b: jnp.ndarray          # (n_static_cells, 4) int32
    surface_edge_mask: jnp.ndarray    # (n_static_cells, 4) bool
    cell_types: jnp.ndarray           # (n_static_cells,) int32
    cell_aabb_min: jnp.ndarray        # (n_static_cells, 2) float32
    cell_aabb_max: jnp.ndarray        # (n_static_cells, 2) float32
    column_cell_indices: jnp.ndarray  # (n_columns, max_cells_per_column) int32
    column_cell_count: jnp.ndarray    # (n_columns,) int32


class ObjectTemplate(NamedTuple):
    """Compiled per-object template used to build object-separated worlds."""

    name: str
    object_id: int
    is_robot: bool
    is_dynamic: bool
    origin_x: int
    origin_y: int
    grid: np.ndarray
    positions: np.ndarray
    masses: np.ndarray
    fixed: np.ndarray
    point_mask: np.ndarray
    spring_a_idx: np.ndarray
    spring_b_idx: np.ndarray
    spring_const: np.ndarray
    spring_rest_length: np.ndarray
    spring_mask: np.ndarray
    rigid_spring_mask: np.ndarray
    boxel_corners: np.ndarray
    boxel_edge_a: np.ndarray
    boxel_edge_b: np.ndarray
    boxel_world_vy: np.ndarray
    boxel_world_vx: np.ndarray
    boxel_types: np.ndarray
    boxel_mask: np.ndarray
    boxel_surface_edge_mask: np.ndarray
    cell_vertices: np.ndarray
    cell_vertex_count: np.ndarray
    cell_edge_a: np.ndarray
    cell_edge_b: np.ndarray
    cell_world_vy: np.ndarray
    cell_world_vx: np.ndarray
    cell_types: np.ndarray
    cell_mask: np.ndarray
    cell_surface_edge_mask: np.ndarray
    robot_point_indices: np.ndarray
    terrain_point_indices: np.ndarray
    terrain_sample_positions: np.ndarray
    actuator_cell_spring_indices: np.ndarray
    actuator_spring_act_count: np.ndarray
    actuated_spring_mask: np.ndarray
    per_axis_h_pairs: np.ndarray
    per_axis_v_pairs: np.ndarray
    per_axis_spring_act_count: np.ndarray
    per_axis_actuated_mask: np.ndarray
    deformation_horiz_springs: np.ndarray
    deformation_vert_springs: np.ndarray
    n_robot_voxels: int
    point_count: int
    spring_count: int


class WorldTemplate(NamedTuple):
    """Compiled world template preserving object boundaries."""

    grid_h: int
    grid_w: int
    robot_name: str
    object_templates: tuple[ObjectTemplate, ...]
    robot_template_index: int


class WorldTemplateSet(NamedTuple):
    """Primary + optional mirrored template set for genome-scale evaluation."""

    primary: WorldTemplate
    mirror: WorldTemplate | None
    mirror_mode: Literal["none", "paired"]


class BuiltWorld(NamedTuple):
    """Concrete built world used by envs and experiment wrappers."""

    sim_state: SimState
    topology: SpringTopology
    constants: PhysicsConstants
    collision_data: CollisionData
    static_collider_data: StaticColliderData
    actuator_info: ActuatorInfo
    per_axis_info: PerAxisActuatorInfo
    deformation_info: DeformationInfo
    robot_point_indices: jnp.ndarray
    robot_point_mask: jnp.ndarray
    dynamic_terrain_point_indices: jnp.ndarray
    fixed_terrain_sample_positions: jnp.ndarray
    render_info: "RenderInfo"
    grid_h: int
    grid_w: int


class SimState(NamedTuple):
    """Full simulation state for a single robot. Vmapped across population."""
    # --- Point state (dynamic, changes every step) ---
    positions: jnp.ndarray            # (n_points, 2) float32
    velocities: jnp.ndarray           # (n_points, 2) float32
    positions_last: jnp.ndarray       # (n_points, 2) — for true velocity
    velocities_true: jnp.ndarray      # (n_points, 2) — finite-difference velocity

    # --- Point properties (static per episode) ---
    masses: jnp.ndarray               # (n_points, 2) float32
    fixed: jnp.ndarray                # (n_points, 2) bool — immovable points
    point_mask: jnp.ndarray           # (n_points,) bool — True=active

    # --- Spring state (dynamic — rest lengths change with actuation) ---
    spring_rest_length: jnp.ndarray        # (n_springs,) float32 — current
    spring_rest_length_goal: jnp.ndarray   # (n_springs,) float32 — actuator target
    spring_init_rest_length: jnp.ndarray   # (n_springs,) float32 — original
    spring_const: jnp.ndarray              # (n_springs,) float32
    spring_mask: jnp.ndarray               # (n_springs,) bool — True=active
    rigid_spring_mask: jnp.ndarray         # (n_springs,) bool — Phase 2 eligibility mask

    # --- External forces (set by collision detection each step) ---
    external_forces: jnp.ndarray      # (n_points, 2) float32
    tangential_deformation: jnp.ndarray  # (n_points, 2) friction-anchor memory
    friction_anchor: jnp.ndarray      # (n_points, 2) world-space PBD anchor
    friction_anchor_active: jnp.ndarray  # (n_points,) bool
    friction_anchor_no_contact_count: jnp.ndarray  # (n_points,) int32
    static_manifold_cell_idx: jnp.ndarray  # (n_points,) int32 legacy static-contact memory id
    static_manifold_edge_idx: jnp.ndarray  # (n_points,) int32 legacy static-contact memory edge id
    static_manifold_active: jnp.ndarray  # (n_points,) bool
    static_manifold_no_contact_count: jnp.ndarray  # (n_points,) int32
    static_manifold_normal_force: jnp.ndarray  # (n_points,) float32 last colliding |N|
    static_manifold_normal: jnp.ndarray  # (n_points, 2) float32 — unit normal from last memory contact


class EnvState(NamedTuple):
    """Environment state for Gym-style interface. Pytree-compatible."""
    sim_state: SimState
    step_count: jnp.ndarray     # () int32
    done: jnp.ndarray           # () bool
    prev_com_x: jnp.ndarray     # () float32 — for reward delta


class ShapeEnvState(NamedTuple):
    """Environment state for shape-changing envs (HeightMaximizer, WingspanMaximizer)."""
    sim_state: SimState
    step_count: jnp.ndarray     # () int32
    done: jnp.ndarray           # () bool
    prev_span: jnp.ndarray      # () float32 — for reward delta


class OrientedEnvState(NamedTuple):
    """Environment state for envs needing orientation (BridgeWalker, UpStepper)."""
    sim_state: SimState
    step_count: jnp.ndarray     # () int32
    done: jnp.ndarray           # () bool
    prev_com_x: jnp.ndarray     # () float32 — for reward delta
    initial_robot_positions: jnp.ndarray  # (n_robot_points, 2) — for orientation


class JumperEnvState(NamedTuple):
    """Environment state for Jumper (tracks both x and y COM)."""
    sim_state: SimState
    step_count: jnp.ndarray     # () int32
    done: jnp.ndarray           # () bool
    prev_com_x: jnp.ndarray     # () float32
    prev_com_y: jnp.ndarray     # () float32


class ClimberEnvState(NamedTuple):
    """Environment state for Climber (tracks COM y for reward)."""
    sim_state: SimState
    step_count: jnp.ndarray     # () int32
    done: jnp.ndarray           # () bool
    prev_com_y: jnp.ndarray     # () float32 — for reward delta


class GridData(NamedTuple):
    """Pre-computed grid infrastructure. Static for a given (H, W).
    Computed once with numpy, reused across all morphologies."""
    positions: jnp.ndarray            # ((H+1)*(W+1), 2) float32
    spring_a_idx: jnp.ndarray         # (n_springs,) int32
    spring_b_idx: jnp.ndarray         # (n_springs,) int32
    spring_rest_length: jnp.ndarray   # (n_springs,) float32
    spring_types: jnp.ndarray         # (n_springs,) int32 — 0=horiz, 1=vert, 2=diag
    spring_cell_a: jnp.ndarray        # (n_springs,) int32 — later cell in row-major (-1 = boundary)
    spring_cell_b: jnp.ndarray        # (n_springs,) int32 — earlier cell (-1 = boundary)
    cell_corner_indices: jnp.ndarray  # (H*W, 4) int32 — [BL, BR, TR, TL]
    cell_grid_vy: jnp.ndarray         # (H*W,) int32
    cell_grid_vx: jnp.ndarray         # (H*W,) int32
    cell_horiz_bottom: jnp.ndarray    # (H*W,) int32
    cell_horiz_top: jnp.ndarray       # (H*W,) int32
    cell_vert_left: jnp.ndarray       # (H*W,) int32
    cell_vert_right: jnp.ndarray      # (H*W,) int32
    cell_diag1: jnp.ndarray           # (H*W,) int32
    cell_diag2: jnp.ndarray           # (H*W,) int32
    H: int
    W: int


class DeformationInfo(NamedTuple):
    """Precomputed voxel→spring mapping for deformation sensing.

    Maps each non-empty robot voxel to its horizontal and vertical edge springs.
    Used to compute per-voxel strain as proprioceptive observation.
    """
    voxel_horiz_springs: jnp.ndarray  # (n_robot_voxels, 2) int32, -1 = missing
    voxel_vert_springs: jnp.ndarray   # (n_robot_voxels, 2) int32, -1 = missing
    n_robot_voxels: int


class RenderInfo(NamedTuple):
    """Static rendering data built at env init. Numpy arrays (not for JIT)."""
    cell_vertices: np.ndarray       # (n_cells, 4) int32 — point indices into dynamic+static positions
    cell_vertex_count: np.ndarray   # (n_cells,) int32 — 3 or 4
    cell_types: np.ndarray          # (n_cells,) int32 — voxel type per cell
    cell_mask: np.ndarray           # (n_cells,) bool — active cells
    surface_edge_mask: np.ndarray   # (n_cells, 4) bool — boundary edges
    cell_is_robot: np.ndarray       # (n_cells,) bool
    cell_is_dynamic: np.ndarray     # (n_cells,) bool — static terrain renders from static points
    cell_object_id: np.ndarray      # (n_cells,) int32 — object ownership/debugging
    robot_point_indices: np.ndarray # (n_robot_points,) int32 — for camera COM
    grid_h: int
    grid_w: int
    actuator_cell_indices: np.ndarray   # (n_act_cells,) int32 — which cells are actuators
    cell_h_spring_pairs: np.ndarray     # (n_cells, 2) int32 — per-render-cell horizontal spring pair, -1=none
    cell_v_spring_pairs: np.ndarray     # (n_cells, 2) int32 — per-render-cell vertical spring pair, -1=none
    spring_init_rest_length: np.ndarray  # (n_springs,) float32 — cached at init
    static_point_positions: np.ndarray  # (n_static_points, 2) float32 — fixed render-only points


class RenderStepOutput(NamedTuple):
    """Per-step output for render-only episode replays."""
    positions: jnp.ndarray          # (n_points, 2)
    spring_rest_length: jnp.ndarray # (n_springs,)
    reward: jnp.ndarray             # ()
    done: jnp.ndarray               # ()


class StepOutput(NamedTuple):
    """Per-step output for lax.scan episode rollouts."""
    obs: jnp.ndarray            # (obs_dim,)
    reward: jnp.ndarray         # ()
    done: jnp.ndarray           # ()
    positions: jnp.ndarray      # (n_points, 2) — for visualization


def default_physics_constants() -> PhysicsConstants:
    """Create the default stable physics constants.

    Most square-cell values are EvoGym-derived. Fields such as ground normal
    damping and flat-contact stiction are documented local extensions. Slope
    contact/grip fields are retained for compatibility, but experimental slope
    terrain is geometry-only in the stable library boundary.
    """
    from . import constants as C
    return PhysicsConstants(
        dt=C.DT,
        gravity=C.GRAVITY,
        viscous_drag=C.VISCOUS_DRAG,
        collision_const_ground=C.COLLISION_CONST_GROUND,
        collision_const_obj=C.COLLISION_CONST_OBJ,
        collision_vel_damping=C.COLLISION_VEL_DAMPING,
        ground_vel_damping=C.GROUND_VEL_DAMPING,
        collision_base_dist=C.COLLISION_BASE_DIST,
        dynamic_friction_const=C.DYNAMIC_FRICTION_CONST,
        friction_const=C.FRICTION_CONST,
        ground_friction_const=C.GROUND_FRICTION_CONST,
        actuator_convergence=C.ACTUATOR_CONVERGENCE,
        integrator=C.INTEGRATOR_RK4,
        static_friction_const=0.0,
        terrain_stiction_stiffness=0.0,
        ground_stiction_stiffness=0.0,
        terrain_stiction_switch_speed=0.0,
        ground_stiction_switch_speed=0.0,
        floor_static_friction_const=0.0,
        floor_stiction_stiffness=0.0,
        floor_stiction_switch_speed=0.0,
        ground_static_friction_const=0.0,
        terrain_stiction_scope=C.STICTION_SCOPE_FLOOR_SLOPE_ONLY,
        friction_model=C.FRICTION_MODEL_NONE,
        anchor_stiffness=C.ANCHOR_STIFFNESS,
        deformation_stiffness=C.DEFORMATION_STIFFNESS,
        deformation_damping=C.DEFORMATION_DAMPING,
        deformation_decay=C.DEFORMATION_DECAY,
        deformation_normal_bleed=C.DEFORMATION_NORMAL_BLEED,
        breakaway_threshold=C.BREAKAWAY_THRESHOLD,
        friction_constraint_mode=C.FRICTION_CONSTRAINT_MODE_OFF,
        friction_anchor_stiffness=C.FRICTION_ANCHOR_STIFFNESS,
        friction_anchor_correction=C.FRICTION_ANCHOR_CORRECTION,
        friction_anchor_persist_substeps=C.FRICTION_ANCHOR_PERSIST_SUBSTEPS,
        slope_contact_mode=C.SLOPE_CONTACT_MODE,
        slope_contact_near_dist=C.SLOPE_CONTACT_NEAR_DIST,
        slope_contact_release_speed=C.SLOPE_CONTACT_RELEASE_SPEED,
        slope_contact_normal_damping=C.SLOPE_CONTACT_NORMAL_DAMPING,
        slope_contact_tangent_damping=C.SLOPE_CONTACT_TANGENT_DAMPING,
        slope_contact_assist_mu=C.SLOPE_CONTACT_ASSIST_MU,
        slope_grip_enabled=C.SLOPE_GRIP_ENABLED,
        slope_grip_static_friction_const=C.SLOPE_GRIP_STATIC_FRICTION_CONST,
        slope_grip_support_stiffness=C.SLOPE_GRIP_SUPPORT_STIFFNESS,
        slope_grip_switch_speed=C.SLOPE_GRIP_SWITCH_SPEED,
        slope_grip_near_contact_dist=C.SLOPE_GRIP_NEAR_CONTACT_DIST,
        slope_grip_near_release_speed=C.SLOPE_GRIP_NEAR_RELEASE_SPEED,
        slope_grip_scope=C.SLOPE_GRIP_SCOPE,
        diagnostics_enabled=True,
    )


def stiction_physics_constants(
    *,
    terrain: bool = False,
    ground: bool = False,
    static_friction_const: float | None = None,
    ground_static_friction_const: float | None = None,
    terrain_stiction_stiffness: float | None = None,
    ground_stiction_stiffness: float | None = None,
    terrain_stiction_switch_speed: float | None = None,
    ground_stiction_switch_speed: float | None = None,
    floor_static_friction_const: float | None = None,
    floor_stiction_stiffness: float | None = None,
    floor_stiction_switch_speed: float | None = None,
    terrain_stiction_scope: int | None = None,
) -> PhysicsConstants:
    """Create PhysicsConstants with optional terrain/ground/floor stiction enabled.

    Terrain and ground stiction are opt-in. Terrain stiction applies to
    non-sloped static edges selected by ``terrain_stiction_scope``; slope-only
    scope is currently inert because experimental slope terrain is
    geometry-only. Floor stiction is independent of ``stiction_enabled`` and
    defaults to disabled.
    """
    from . import constants as C

    mu_s = (
        float(C.STATIC_FRICTION_CONST)
        if static_friction_const is None
        else float(static_friction_const)
    )
    ground_mu_s = (
        mu_s
        if ground_static_friction_const is None
        else float(ground_static_friction_const)
    )
    terrain_k = (
        float(C.TERRAIN_STICTION_STIFFNESS)
        if terrain_stiction_stiffness is None
        else float(terrain_stiction_stiffness)
    )
    ground_k = (
        float(C.GROUND_STICTION_STIFFNESS)
        if ground_stiction_stiffness is None
        else float(ground_stiction_stiffness)
    )
    terrain_switch = (
        float(C.TERRAIN_STICTION_SWITCH_SPEED)
        if terrain_stiction_switch_speed is None
        else float(terrain_stiction_switch_speed)
    )
    ground_switch = (
        float(C.GROUND_STICTION_SWITCH_SPEED)
        if ground_stiction_switch_speed is None
        else float(ground_stiction_switch_speed)
    )
    floor_mu_s = (
        0.0
        if floor_static_friction_const is None
        else float(floor_static_friction_const)
    )
    floor_k = (
        0.0
        if floor_stiction_stiffness is None
        else float(floor_stiction_stiffness)
    )
    floor_switch = (
        0.0
        if floor_stiction_switch_speed is None
        else float(floor_stiction_switch_speed)
    )
    scope = (
        int(C.STICTION_SCOPE_FLOOR_SLOPE_ONLY)
        if terrain_stiction_scope is None
        else int(terrain_stiction_scope)
    )
    if not terrain and not ground:
        mu_s = 0.0
    return default_physics_constants()._replace(
        stiction_enabled=bool(terrain or ground),
        static_friction_const=mu_s,
        terrain_stiction_stiffness=(terrain_k if terrain else 0.0),
        ground_stiction_stiffness=(ground_k if ground else 0.0),
        terrain_stiction_switch_speed=(terrain_switch if terrain else 0.0),
        ground_stiction_switch_speed=(ground_switch if ground else 0.0),
        floor_static_friction_const=floor_mu_s,
        floor_stiction_stiffness=floor_k,
        floor_stiction_switch_speed=floor_switch,
        ground_static_friction_const=(ground_mu_s if ground else 0.0),
        terrain_stiction_scope=scope,
        friction_model=C.FRICTION_MODEL_NONE,
        anchor_stiffness=C.ANCHOR_STIFFNESS,
        deformation_stiffness=C.DEFORMATION_STIFFNESS,
        deformation_damping=C.DEFORMATION_DAMPING,
        deformation_decay=C.DEFORMATION_DECAY,
        deformation_normal_bleed=C.DEFORMATION_NORMAL_BLEED,
        breakaway_threshold=C.BREAKAWAY_THRESHOLD,
        friction_constraint_mode=C.FRICTION_CONSTRAINT_MODE_OFF,
        friction_anchor_stiffness=C.FRICTION_ANCHOR_STIFFNESS,
        friction_anchor_correction=C.FRICTION_ANCHOR_CORRECTION,
        friction_anchor_persist_substeps=C.FRICTION_ANCHOR_PERSIST_SUBSTEPS,
    )
