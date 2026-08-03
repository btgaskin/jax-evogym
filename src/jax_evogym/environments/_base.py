"""Base environment class for object-separated JAX EvoGym environments."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..build import compile_world_templates, instantiate_world
from ..data import data_path
from ..world import EvoWorld


class EvoGymBaseEnv:
    """Base class for all JAX EvoGym environments."""

    def __init__(
        self,
        json_name: str,
        body: np.ndarray,
        spawn_x: int,
        spawn_y: int,
        connections: np.ndarray | None = None,
        max_steps: int = 500,
    ):
        world = EvoWorld.from_json(data_path(json_name))
        world.add_from_array("robot", body, spawn_x, spawn_y, connections=connections)

        template_set = compile_world_templates(world, robot_name="robot", mirror_mode="none")
        built = instantiate_world(template_set.primary)

        self._init_sim_state = built.sim_state
        self.topology = built.topology
        self.constants = built.constants
        self.actuator_info = built.actuator_info
        self.per_axis_info = built.per_axis_info
        self.collision_data = built.collision_data
        self.static_collider_data = built.static_collider_data
        self.deformation_info = built.deformation_info
        self.robot_point_indices = built.robot_point_indices
        self.robot_point_mask = built.robot_point_mask
        self.dynamic_terrain_point_indices = built.dynamic_terrain_point_indices
        self.max_steps = max_steps

        point_mask_np = np.array(built.sim_state.point_mask)
        self.active_point_indices = jnp.array(np.where(point_mask_np)[0], dtype=jnp.int32)

        fixed_samples = np.array(built.fixed_terrain_sample_positions)
        if fixed_samples.size == 0:
            self.terrain_positions = jnp.zeros((1, 2), dtype=jnp.float32)
            self.terrain_mask = jnp.zeros((1,), dtype=jnp.bool_)
        else:
            self.terrain_positions = jnp.array(fixed_samples, dtype=jnp.float32)
            self.terrain_mask = jnp.ones((fixed_samples.shape[0],), dtype=jnp.bool_)

        self.n_robot_points = int(self.robot_point_indices.shape[0])
        self.n_robot_voxels = int(self.deformation_info.n_robot_voxels)
        self.n_active = int(self.active_point_indices.shape[0])
        self.n_actuators = int(self.actuator_info.cell_spring_indices.shape[0])
        self.n_points = int(built.sim_state.positions.shape[0])
        self.render_info = built.render_info
