from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from foos.assets_cfg.ball import BALL_CFG
from foos.assets_cfg.foosball_table import FOOSBALL_TABLE_CFG

# Mapping from rod link name to the pre-converted player-figure USD mesh.
# The Isaac Sim 5.1 URDF importer crashes on the rod OBJ files in jointed
# contexts (see CLAUDE.md), so the URDF carries cylinder primitives for
# physics+visual and we attach the real player mesh as a USD child prim
# under each rod link at scene-setup time.
#
# Each entry is (mesh_subdir, team_color). Team 1 = red, team 2 = blue.
_ROD_TO_MESH: dict[str, tuple[str, tuple[float, float, float]]] = {
    "goalie_team_1": ("goalie_centered_team_1", (0.8, 0.2, 0.2)),
    "bar2_team_1":   ("def_centered_team_1",    (0.8, 0.2, 0.2)),
    "bar5_team_1":   ("mid_centered_team_1",    (0.8, 0.2, 0.2)),
    "bar3_team_1":   ("att_centered_team_1",    (0.8, 0.2, 0.2)),
    "goalie_team_2": ("goalie_centered_team_2", (0.2, 0.2, 0.8)),
    "bar2_team_2":   ("def_centered_team_2",    (0.2, 0.2, 0.8)),
    "bar5_team_2":   ("mid_centered_team_2",    (0.2, 0.2, 0.8)),
    "bar3_team_2":   ("att_centered_team_2",    (0.2, 0.2, 0.8)),
}
_ROD_MESH_DIR = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "foosball_table"
    / "meshes"
    / "usd"
)


@configclass
class FoosEnvCfg(DirectRLEnvCfg):
    # 1/240 keeps ball-rod contacts stable; render every 4 sim steps.
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 240.0, render_interval=4)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5)

    decimation = 4
    episode_length_s = 30.0

    # 16 joints = 8 prismatic (slide) + 8 revolute (spin), one pair per rod.
    action_space = 16
    # 16 q + 16 qdot + 3 ball pos + 3 ball lin_vel + 1 phase placeholder = 39
    observation_space = 39
    state_space = 0

    robot: object = FOOSBALL_TABLE_CFG.replace(
        prim_path="/World/envs/env_.*/Table"
    )
    ball: object = BALL_CFG.replace(prim_path="/World/envs/env_.*/Ball")

    # Action scaling, applied per joint type. TUNABLE.
    action_scale_prismatic: float = 0.15  # meters
    action_scale_revolute: float = 1.5    # radians

    # Elevated side view, pitched down toward the playing surface.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 1.4, 2.4),
        lookat=(0.0, 0.0, 0.85),
    )

    # Ball-out-of-play bounds in env-local frame. Resetting when the ball
    # leaves these limits keeps episodes finite once we wire up rewards.
    # Table footprint: |x| ~= 0.78, |y| ~= 0.40; field surface ~= z=0.84.
    ball_x_limit: float = 0.7
    ball_y_limit: float = 0.42
    ball_z_min: float = 0.5


class FoosEnv(DirectRLEnv):
    cfg: FoosEnvCfg

    def __init__(self, cfg: FoosEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Build per-joint scale tensor (16,) by dispatching on joint name.
        joint_names = self.table.joint_names
        scales = torch.zeros(len(joint_names), device=self.device)
        for i, name in enumerate(joint_names):
            if name.endswith("_prismatic_joint"):
                scales[i] = self.cfg.action_scale_prismatic
            elif name.endswith("_revolute_joint"):
                scales[i] = self.cfg.action_scale_revolute
            else:
                raise ValueError(f"Unexpected joint name in foosball URDF: {name!r}")
        self._joint_scale = scales

        # Default joint position (zeros — rods centered, players upright).
        self._joint_default = torch.zeros(
            (self.num_envs, len(joint_names)), device=self.device
        )

        self._action_targets = self._joint_default.clone()
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)

    def _setup_scene(self):
        self.table = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)

        # Attach the rod-figure USD meshes under each rod link's source-env
        # prim path. clone_environments below will replicate them across envs.
        for link_name, (subdir, color) in _ROD_TO_MESH.items():
            usd_path = _ROD_MESH_DIR / subdir / f"{subdir}.usd"
            mesh_cfg = sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            mesh_cfg.func(
                prim_path=f"/World/envs/env_0/Table/{link_name}/figure",
                cfg=mesh_cfg,
            )

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 1.0))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["table"] = self.table
        self.scene.rigid_objects["ball"] = self.ball

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions in [-1, 1]; offset from neutral pose by per-joint scale.
        self._action_targets = self._joint_default + actions.clamp(-1.0, 1.0) * self._joint_scale

    def _apply_action(self) -> None:
        self.table.set_joint_position_target(self._action_targets)

    def _get_observations(self) -> dict:
        joint_pos = self.table.data.joint_pos
        joint_vel = self.table.data.joint_vel

        # Ball pose/velocity in env-local frame (subtract env origin).
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        ball_lin_vel = self.ball.data.root_lin_vel_w

        obs = torch.cat(
            [joint_pos, joint_vel, ball_pos, ball_lin_vel, self._phase],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # Terminate if the ball has left the playing field.
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        out_of_bounds = (
            (ball_pos[:, 0].abs() > self.cfg.ball_x_limit)
            | (ball_pos[:, 1].abs() > self.cfg.ball_y_limit)
            | (ball_pos[:, 2] < self.cfg.ball_z_min)
        )
        return out_of_bounds, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.table._ALL_INDICES

        super()._reset_idx(env_ids)

        # Zero the rods and clear velocities.
        joint_pos = self._joint_default[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        self.table.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Drop ball at table center with a small random XY jitter.
        n = len(env_ids)
        jitter = torch.empty(n, 2, device=self.device).uniform_(-0.05, 0.05)
        ball_root_state = self.ball.data.default_root_state[env_ids].clone()
        ball_root_state[:, :3] += self.scene.env_origins[env_ids]
        ball_root_state[:, 0:2] += jitter
        ball_root_state[:, 7:13] = 0.0  # zero linear + angular velocity
        self.ball.write_root_pose_to_sim(ball_root_state[:, :7], env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(ball_root_state[:, 7:], env_ids=env_ids)

        self._phase[env_ids] = 0.0
