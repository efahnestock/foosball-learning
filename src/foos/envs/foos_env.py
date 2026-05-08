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
    # 16 q + 16 qdot + 3 ball pos + 3 ball lin_vel = 38
    observation_space = 38
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
    # A goal counts only when the ball is past the goalie X-line *and*
    # inside the narrow goal mouth in Y. Anything past X but outside the
    # mouth (e.g. ball flying off the side of the table) is OOB, not a goal.
    ball_x_limit: float = 0.7              # past goalie line
    ball_y_limit: float = 0.42             # off-side bound
    ball_z_min: float = 0.5                # fell-through bound
    goal_mouth_half_width: float = 0.10    # half-width of the goal opening (m)

    # Reward weights for the "score any goal, fast, smoothly" objective.
    # Directional shaping only (|ball.vx|) — generic ball speed produced
    # rod-spinners that gamed the dense signal. Action-delta penalty is the
    # real lever for non-twitchy motion; magnitude cost just keeps rods off
    # the joint stops. OOB is set on the order of the goal bonus so kicking
    # the ball off the side hurts as much as a goal helps.
    rew_scale_ball_speed: float = 0.0          # off — replaced by directed below
    rew_scale_ball_x_speed: float = 0.0        # off — replaced by directed below
    rew_scale_goal_proximity: float = 0.0      # off — proximity-only is a trap:
                                               # the policy parked the ball near a
                                               # mouth and milked the bonus instead
                                               # of scoring (zero actual goals)
    # x-velocity *toward* the nearest goal mouth, clipped at zero.
    # Positive only when the ball is actively being driven into a goal —
    # parking earns nothing, motion away earns nothing. This was the missing
    # piece: previous shapings either rewarded any motion (rod-spinners) or
    # static proximity (ball-parkers).
    rew_scale_v_toward_goal: float = 0.1
    rew_scale_action: float = -5.0e-3          # mild magnitude cost (smoothness
                                               # below is the real anti-twitch lever;
                                               # raising magnitude cost any further
                                               # collapses exploration to "rods idle")
    rew_scale_action_delta: float = -1.0e-3    # mild; LPF below is the real
                                               # smoothness mechanism, the reward
                                               # penalty doubles up too much when
                                               # set higher — the policy goes idle
    rew_scale_step: float = -0.02              # softer time pressure (was -0.05)
                                               # so idle isn't disproportionately
                                               # cheaper than risky scoring
    rew_scale_goal: float = 25.0               # bigger payoff so scoring beats idle
                                               # by enough margin to overcome the
                                               # off-mouth-x risk on the way
    rew_scale_oob: float = -3.0                # gentler than -5 so the policy
                                               # is willing to risk a kick attempt;
                                               # also includes "past goalie x but
                                               # outside goal mouth" (back corners)

    # Action low-pass filter (per-step blend factor). Action targets are
    # smoothed via a first-order IIR before being sent to PhysX:
    #   target := alpha * new + (1 - alpha) * prev_target
    # alpha=0.35 at 60 Hz => ~50 ms time constant. 0.2 was too laggy for
    # kicking — the action target couldn't snap fast enough to form a kick
    # before the ball moved past, and the policy converged on idle.
    action_smoothing_alpha: float = 0.35


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
        # `_last_actions` tracks the action commanded *this* step (after clip);
        # `_prev_actions` tracks the previous step, used to compute the
        # smoothness delta penalty.
        self._last_actions = torch.zeros_like(self._action_targets)
        self._prev_actions = torch.zeros_like(self._action_targets)

        # Per-step termination flags, set in _get_dones and consumed by
        # _get_rewards in the same step (DirectRLEnv runs dones before rewards).
        # We still track per-side goals separately (handy for future self-play
        # logging) but the reward only sees `_goal_any`.
        self._goal_team1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_team2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._oob = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Cumulative goal counter for tensorboard logging. Per-team breakdown
        # is kept too so we can spot if the policy only scores in one direction.
        self._score_team1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._score_team2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
        # Then low-pass filter the target so PD doesn't snap from one pose to
        # the next on every step.
        clipped = actions.clamp(-1.0, 1.0)
        target_raw = self._joint_default + clipped * self._joint_scale
        alpha = self.cfg.action_smoothing_alpha
        self._action_targets = alpha * target_raw + (1.0 - alpha) * self._action_targets
        self._prev_actions = self._last_actions
        self._last_actions = clipped

    def _apply_action(self) -> None:
        self.table.set_joint_position_target(self._action_targets)

    def _get_observations(self) -> dict:
        joint_pos = self.table.data.joint_pos
        joint_vel = self.table.data.joint_vel

        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        ball_lin_vel = self.ball.data.root_lin_vel_w

        obs = torch.cat([joint_pos, joint_vel, ball_pos, ball_lin_vel], dim=-1)
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        bx, by, bz = ball_pos[:, 0], ball_pos[:, 1], ball_pos[:, 2]

        # A goal requires both: ball past the goalie X line AND inside the
        # narrow goal mouth in Y. Past X but off-mouth (ball flying off the
        # back corners) is OOB, not a goal — without this Y check, training
        # rewards "ball crossed the goal line anywhere on the table edge".
        in_mouth = by.abs() < self.cfg.goal_mouth_half_width
        past_x_neg = bx < -self.cfg.ball_x_limit
        past_x_pos = bx > self.cfg.ball_x_limit
        self._goal_team1 = past_x_neg & in_mouth   # ball into team 2's net
        self._goal_team2 = past_x_pos & in_mouth   # ball into team 1's net
        self._goal_any = self._goal_team1 | self._goal_team2
        off_mouth_x = (past_x_neg | past_x_pos) & ~in_mouth
        self._oob = (
            (by.abs() > self.cfg.ball_y_limit)
            | (bz < self.cfg.ball_z_min)
            | off_mouth_x
        )

        terminated = self._goal_any | self._oob

        # Update cumulative score counters before reset wipes them.
        self._score_team1 += self._goal_team1.long()
        self._score_team2 += self._goal_team2.long()

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        ball_pos_local = self.ball.data.root_pos_w - self.scene.env_origins
        ball_lin_vel = self.ball.data.root_lin_vel_w
        ball_speed = ball_lin_vel[:, :2].norm(dim=-1)
        ball_x_speed = ball_lin_vel[:, 0].abs()
        bx = ball_pos_local[:, 0]

        # Signed velocity toward the nearest goal mouth. Positive when ball
        # heads outward (toward closer net), negative when it heads inward.
        # IMPORTANT: do NOT clamp at zero — without the clamp this is a
        # proper potential-based shaping (sum over episode telescopes to
        # |x_final| - |x_initial|), so oscillating the ball nets zero and
        # the policy can't farm reward by pushing-then-pulling. With the
        # clamp, the agent can repeatedly push forward, take negative motion
        # for free, and accumulate large fake rewards without ever scoring.
        sign_to_nearest = torch.sign(bx)  # closer goal is on the same side
        v_toward = ball_lin_vel[:, 0] * sign_to_nearest

        action_cost = self._last_actions.square().sum(dim=-1)
        action_delta = (self._last_actions - self._prev_actions).square().sum(dim=-1)
        step_penalty = torch.full(
            (self.num_envs,), cfg.rew_scale_step, device=self.device
        )

        rew = (
            cfg.rew_scale_ball_speed * ball_speed
            + cfg.rew_scale_ball_x_speed * ball_x_speed
            + cfg.rew_scale_v_toward_goal * v_toward
            + cfg.rew_scale_action * action_cost
            + cfg.rew_scale_action_delta * action_delta
            + step_penalty
            + cfg.rew_scale_goal * self._goal_any.float()
            + cfg.rew_scale_oob * self._oob.float()
        )

        # Surface per-component means in extras for tensorboard.
        self.extras["log"] = {
            "rew/ball_speed": (cfg.rew_scale_ball_speed * ball_speed).mean(),
            "rew/ball_x_speed": (cfg.rew_scale_ball_x_speed * ball_x_speed).mean(),
            "rew/v_toward_goal": (cfg.rew_scale_v_toward_goal * v_toward).mean(),
            "rew/action_cost": (cfg.rew_scale_action * action_cost).mean(),
            "rew/action_delta": (cfg.rew_scale_action_delta * action_delta).mean(),
            "rew/step": step_penalty.mean(),
            "rew/goal_any": (cfg.rew_scale_goal * self._goal_any.float()).mean(),
            "rew/oob": (cfg.rew_scale_oob * self._oob.float()).mean(),
            "score/goals_total": (self._score_team1 + self._score_team2).float().mean(),
            "score/team1_total": self._score_team1.float().mean(),
            "score/team2_total": self._score_team2.float().mean(),
        }
        return rew

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

        self._last_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        # Re-anchor the low-pass-filtered targets at the neutral pose so the
        # next episode doesn't inherit the previous one's smoothed trajectory.
        self._action_targets[env_ids] = self._joint_default[env_ids]
