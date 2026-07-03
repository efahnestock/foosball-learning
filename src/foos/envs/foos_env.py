from __future__ import annotations

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import (
    ConvexDecompositionPropertiesCfg,
    SDFMeshPropertiesCfg,
    define_mesh_collision_properties,
)
from isaaclab.utils import configclass

from foos.assets_cfg.ball import BALL_CFG
from foos.assets_cfg.foosball_table import FOOSBALL_TABLE_CFG

# NOTE: `OpponentActor` (self-play) is imported lazily inside set_opponent()
# so the env — and the whole `foos` package — imports without the self-play
# module present. Only needed when cfg.self_play_mode is enabled.

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
    # In single-agent mode (self_play_mode=False, the default) one policy
    # controls all 16. In self-play mode the trainee controls team 1's 8
    # joints and an embedded opponent controls team 2's 8 — set
    # action_space=8 externally when flipping self_play_mode on.
    action_space = 16
    # 16 q + 16 qdot + 3 ball pos + 3 ball lin_vel = 38.
    observation_space = 38
    state_space = 0

    # Self-play (EXPERIMENTAL / off by default): when True, action_space
    # must be set to 8, and cfg.opponent_checkpoint must point at an
    # rl_games .pth that drives team 2. Reward switches to zero-sum on
    # goals (team-1-perspective).
    self_play_mode: bool = False
    # Path to an rl_games .pth checkpoint to drive team 2. Ignored unless
    # self_play_mode=True. None = opponent rods stay at their joint defaults.
    opponent_checkpoint: str | None = None
    # When True, opponent uses the deterministic mean (mu) of its policy.
    opponent_deterministic: bool = True

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
    # Playing field is 1.2 m × 0.68 m → half-extents 0.6 × 0.34. Back walls
    # at x = ±0.605 (0.005 past field edge); ball_x_limit slightly past that.
    ball_x_limit: float = 0.62             # past goalie line / through back wall
    ball_y_limit: float = 0.4              # off-side bound (field y-half = 0.34)
    ball_z_min: float = 0.5                # fell-through bound
    # 0.22 m wide × 0.08 m tall goal opening. With field z=0.762 and wall
    # top z=0.912, opening spans z=0.762..0.842 and the cap spans z=0.842..0.912
    # (0.07 m cap). 0.842 is also where the visual-mesh inner-rim plateau
    # sits, so the cap visually aligns with the rim.
    goal_mouth_half_width: float = 0.11    # half-width of the goal opening (m)
    goal_height: float = 0.08              # vertical span of the goal opening (m)
    # Z range the ball must be in for a goal to count. Without this check,
    # a ball that spawns above the table and flies *past* the goalie x-line
    # registers as a goal mid-air without ever rolling on the surface.
    # Field surface z=0.762, ball-at-rest center ≈ 0.779. Wall top z=0.912,
    # goal-opening top z=0.842. Range covers "ball anywhere inside the
    # playing-field cavity".
    goal_z_min: float = 0.74
    goal_z_max: float = 0.92

    # Stuck-ball detection. Real foosball dead-balls (ball wedged in a
    # corner where no rod reaches, or come to rest in a mid-field dead
    # zone) would otherwise burn the whole 30 s episode with no play. If
    # the ball's planar (xy) speed stays below `stuck_speed_threshold`
    # (m/s) for `stuck_time_s` continuous seconds, the point is declared
    # dead: the episode ends (grouped with truncation, so it's a no-score
    # reset and the value function bootstraps rather than eating a
    # spurious terminal penalty). Set `stuck_time_s <= 0` to disable.
    stuck_speed_threshold: float = 0.05
    stuck_time_s: float = 3.0

    # Print a per-step ball-state heartbeat line from _get_dones. Very
    # noisy — meant for physics debugging only. Independent of
    # `log_terminations` (which prints one line per goal/OOB event).
    debug_heartbeat: bool = False

    # Reward weights for the self-play scoring objective. Zero-sum on goals:
    # team 1 scoring = +rew_scale_goal for the trainee, team 2 scoring =
    # -rew_scale_goal. Trainee attacks toward -X (team 2's net at x=-0.6);
    # v_toward_goal directly rewards leftward ball velocity.
    rew_scale_ball_speed: float = 0.0          # off — directed shaping below
    rew_scale_ball_x_speed: float = 0.0        # off
    rew_scale_goal_proximity: float = 0.0      # off
    rew_scale_v_toward_goal: float = 0.5       # +0.5 * (-vx) per step.
                                               # Potential-based: telescopes
                                               # to 0.5 * (x_init - x_final),
                                               # so oscillating ball nets 0.
    rew_scale_action: float = 0.0              # off
    rew_scale_action_delta: float = 0.0        # off until policy reliably
                                               # scores; layer in later for
                                               # smoothness.
    rew_scale_step: float = -0.01              # mild time pressure: idle
                                               # over 1800 steps costs -18,
                                               # one goal +100 dominates.
    rew_scale_goal: float = 100.0              # big terminal bonus, zero-sum
    rew_scale_oob: float = -10.0               # OOB hurts both teams (episode
                                               # ends without a goal). Penalty
                                               # smaller than goal bonus so
                                               # ball-control attempts that
                                               # occasionally fail are still
                                               # net-positive.

    # Action low-pass filter (per-step blend factor). Action targets are
    # smoothed via a first-order IIR before being sent to PhysX:
    #   target := alpha * new + (1 - alpha) * prev_target
    # alpha=1.0 disables the filter (pure pass-through). Was 0.35 (50 ms
    # time constant) for human-like reaction times, but during training
    # this muted rod swings too much — agent couldn't actually kick the
    # ball within a few steps. Re-enable as a post-training smoothness
    # constraint if needed.
    action_smoothing_alpha: float = 1.0

    # Diagnostic: print ball position at every termination event. Off by
    # default — flip on for debugging "are goals real or back-wall hits".
    log_terminations: bool = False

    # Override the table's mesh collision approximation post-spawn. The URDF
    # importer's `collider_type="convex_decomposition"` (set in
    # foosball_table.py) ships only with default decomposition params, which
    # in our case fails to capture the goal-mouth carveout — the ball gets
    # ejected from the closed convex hulls instead of entering the goal.
    # Options:
    #   "default":              keep whatever the URDF importer produced
    #   "convex_decomposition": apply ConvexDecompositionPropertiesCfg with
    #                           aggressive params (more hulls, finer voxels)
    #   "sdf":                  use signed-distance-field mesh collision —
    #                           exact but ~2-3x slower physics
    # Note: both fail empirically because the OBJ mesh is an open shell.
    # Use `use_primitive_table_collision=True` instead.
    table_collision_approx: str = "default"

    # Disable the URDF table's mesh collision and replace it with hand-built
    # primitive walls + floor. The OBJ mesh isn't watertight — convex_decomp
    # closes the playing-field cavity and SDF can't compute a proper inside
    # vs outside, so the ball gets ejected violently from the table interior.
    # Primitives sidestep this entirely: floor + 4 side walls + 2 back-wall
    # pairs (each pair flanking a goal-mouth gap of width 2*goal_mouth_half_width).
    # Defaults to True now that the URDF's table <collision> block has been
    # stripped — without primitives, the table has no collision at all.
    use_primitive_table_collision: bool = True

    # When True, render the primitive cuboid colliders as bright translucent
    # green so the collision shape is visually inspectable. Only meant for
    # debugging the collision geometry — leave False for training.
    visualize_primitive_collision: bool = False

    # Curriculum: at reset, sample ball X position from
    #   sign * Uniform(spawn_x_near, spawn_x_far)
    # with sign +/-1 chosen uniformly. Stage-1 bootstrap: spawn the ball
    # past the goalie (x=±0.525) with an initial velocity into the goal,
    # so a do-nothing policy already scores most episodes — gives the
    # value network a real signal that "kicking the ball outward = +100".
    # Once scoring is reliable, lower spawn_x_near and ball_spawn_vx
    # toward zero to widen the start distribution.
    # Curriculum-decayed spawn. Effective spawn at training step t blends
    # between the easy bootstrap (ball past the goalie at x=±0.55..0.59,
    # vx=0.5) and the hard target (ball at the center, no push) over
    # `curriculum_decay_steps` *frames* of training. _reset_idx samples
    # from the interpolated range every reset.
    # Ball spawns stationary at a random x in ±[near, far] with random sign,
    # near zero velocity. With prismatic working, the agent can chase from
    # any starting position. Random sign means trainee sees both attacking
    # and defending starts.
    ball_spawn_x_near: float = 0.0
    ball_spawn_x_far: float = 0.3
    ball_spawn_vx_toward_goal: float = 0.0
    ball_spawn_random_vmax: float = 0.0
    # Target distribution centered around the goalie line. Pinned here
    # (not at x=0) because the policy collapsed past progress=0.5 in
    # earlier runs — center-spawn requires multi-rod coordination that
    # the value net hasn't bootstrapped. Once the agent reliably scores
    # from this range, lower the target gradually.
    curriculum_target_x_near: float = 0.275
    curriculum_target_x_far: float = 0.55
    curriculum_target_vx: float = 0.25
    # Per-env env-steps over which the curriculum interpolates from
    # bootstrap to target. self.common_step_counter ticks once per
    # env-step (not per env-frame), so 48000 ≈ 1500 PPO epochs at
    # horizon=32. Set huge to effectively disable (always bootstrap).
    curriculum_decay_steps: int = 10**12


class FoosEnv(DirectRLEnv):
    cfg: FoosEnvCfg

    def __init__(self, cfg: FoosEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Build per-joint scale tensor (16,) by dispatching on joint name.
        joint_names = self.table.joint_names
        scales = torch.zeros(len(joint_names), device=self.device)
        team1_idx: list[int] = []
        team2_idx: list[int] = []
        for i, name in enumerate(joint_names):
            if name.endswith("_prismatic_joint"):
                scales[i] = self.cfg.action_scale_prismatic
            elif name.endswith("_revolute_joint"):
                scales[i] = self.cfg.action_scale_revolute
            else:
                raise ValueError(f"Unexpected joint name in foosball URDF: {name!r}")
            if "_team_1_" in name:
                team1_idx.append(i)
            elif "_team_2_" in name:
                team2_idx.append(i)
            else:
                raise ValueError(f"Joint name lacks team marker: {name!r}")
        if len(team1_idx) != 8 or len(team2_idx) != 8:
            raise ValueError(
                f"Expected 8 joints per team, got team1={len(team1_idx)}, "
                f"team2={len(team2_idx)} from joint_names={joint_names!r}"
            )
        self._joint_scale = scales
        self._team1_idx = torch.tensor(team1_idx, dtype=torch.long, device=self.device)
        self._team2_idx = torch.tensor(team2_idx, dtype=torch.long, device=self.device)
        # Per-team action-scale slices (size 8 each) so trainee action[i]
        # maps to joint scales[team1_idx[i]].
        self._joint_scale_team1 = scales[self._team1_idx]
        self._joint_scale_team2 = scales[self._team2_idx]

        # Default joint position (zeros — rods centered, players upright).
        self._joint_default = torch.zeros(
            (self.num_envs, len(joint_names)), device=self.device
        )

        self._action_targets = self._joint_default.clone()
        # Cached 16-wide action history: even though trainee only emits 8
        # values, we track full-16 for smoothness/diagnostic logging and so
        # the action_delta penalty (when enabled) sees both teams' commands.
        self._last_actions = torch.zeros_like(self._action_targets)
        self._prev_actions = torch.zeros_like(self._action_targets)

        # Opponent slot. Only used when cfg.self_play_mode=True; in
        # single-agent mode both teams are controlled by one policy so no
        # opponent is loaded. Cached obs tensor is fed to the opponent at
        # the *next* _pre_physics_step.
        self._opponent_actor: OpponentActor | None = None
        self._last_observation: torch.Tensor | None = None

        # Per-step termination flags, set in _get_dones and consumed by
        # _get_rewards in the same step (DirectRLEnv runs dones before rewards).
        # Per-side goal flags are also used by the self-play zero-sum reward.
        self._goal_team1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_team2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._oob = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Cumulative goal counter for tensorboard logging. Per-team breakdown
        # lets you spot if the policy only scores in one direction.
        self._score_team1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._score_team2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Stuck-ball detection: consecutive-step counter of low ball speed,
        # plus the per-step "stuck this step" flag (grouped into truncated).
        # `_stuck_steps` converts stuck_time_s into policy steps; the policy
        # timestep is sim.dt * decimation (== 1/60 s with the defaults).
        self._stuck_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stuck = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stuck_total = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        policy_dt = self.cfg.sim.dt * self.cfg.decimation
        self._stuck_steps = (
            int(round(self.cfg.stuck_time_s / policy_dt))
            if self.cfg.stuck_time_s > 0
            else 0
        )

        if self.cfg.self_play_mode and self.cfg.opponent_checkpoint is not None:
            self.set_opponent(self.cfg.opponent_checkpoint)

    def set_opponent(self, checkpoint_path: str | None) -> None:
        """Load (or clear) the team-2 opponent policy. Called at env init
        and again whenever the snapshot rotates (Phase 3)."""
        if checkpoint_path is None:
            self._opponent_actor = None
            return
        from foos.envs.opponent_actor import OpponentActor

        self._opponent_actor = OpponentActor.from_checkpoint(
            path=checkpoint_path,
            obs_dim=self.cfg.observation_space,
            team1_idx=self._team1_idx,
            team2_idx=self._team2_idx,
            device=self.device,
        )

    def _setup_scene(self):
        self.table = Articulation(self.cfg.robot)
        self.ball = RigidObject(self.cfg.ball)

        # Override the table's mesh collision approximation if requested.
        # The URDF importer cooks one approximation at import time, but we
        # can re-apply with different params or switch to SDF after spawn.
        approx = self.cfg.table_collision_approx
        if approx != "default":
            if approx == "convex_decomposition":
                mesh_cfg = ConvexDecompositionPropertiesCfg(
                    max_convex_hulls=512,
                    hull_vertex_limit=128,
                    voxel_resolution=2_000_000,
                    error_percentage=2.0,
                    shrink_wrap=True,
                )
            elif approx == "sdf":
                mesh_cfg = SDFMeshPropertiesCfg(
                    sdf_resolution=512,
                )
            else:
                raise ValueError(
                    f"Unknown table_collision_approx={approx!r}; expected "
                    "one of 'default', 'convex_decomposition', 'sdf'."
                )
            # The decorator @apply_nested on define_mesh_collision_properties
            # recurses into all child mesh prims under the given path, so
            # passing the table's source-env prim path covers every collision
            # mesh in the URDF (table body + intermediate empty links).
            print(
                f">>> [foos_env] applying table_collision_approx={approx!r} "
                "to /World/envs/env_0/Table",
                flush=True,
            )
            define_mesh_collision_properties(
                prim_path="/World/envs/env_0/Table", cfg=mesh_cfg
            )

        if self.cfg.use_primitive_table_collision:
            self._build_primitive_table_collision()

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

    def _build_primitive_table_collision(self) -> None:
        """Spawn 9 primitive cuboid colliders for the playing-field cavity.
        The visible OBJ mesh stays for rendering; only the *physics*
        collision changes. Coordinates in env-local frame (env origin at
        world (0,0,0)).

        Real foosball-table geometry (anchored to visible mesh):
            * playing field         1.2 m × 0.68 m  (half-extents 0.6 × 0.34)
            * field surface z       0.732
            * goal-opening top z    0.842 (matches visible mesh)
            * wall top z            0.912 (mesh top after 0.482 m lift)
            * goal opening          0.22 m wide × 0.11 m tall
            * wall total            0.18 m (= 0.912 − 0.732)

        Layout (9 boxes):
            1 floor + 2 long side walls + 4 back-wall segments flanking each
            goal mouth + 2 goal roofs capping the gap above each opening.

        Note: tried bundling cuboids into one composite USD with cooked
        convex_decomposition. The composite collision didn't activate when
        spawned via UsdFileCfg (ball fell through), and at the same hull
        count there was no FPS win. Per-prim cuboids stay.
        """
        import isaaclab.sim as sim_utils

        # Spawn under the env root, NOT under Table — the Table prim has a
        # +0.482 m world-z translation (its init_state.pos), and children
        # inherit that. Spawning here keeps coordinates in env-local frame.
        base = "/World/envs/env_0/primitive_collision"

        FIELD_X = 0.6
        FIELD_Y = 0.34
        # Real-table geometry (per user spec): feet on ground (z=0), field
        # surface 0.762 m above, wall top 0.15 m above field (z=0.912 —
        # matches the visual-mesh top). The visual mesh's playing-field
        # plateau is incorrectly at z=0.842 (0.08 m too high); collision is
        # at the *physical* z=0.762 and ball will visually sit below the
        # mesh plateau until the OBJ gets re-authored.
        FIELD_Z = 0.762
        # Wall height splits in two: side walls (y=±FIELD_Y) MUST stay below
        # the rods at z=0.842 (rod radius 0.008 → bottom 0.834), otherwise
        # the rod's cylinder collides with the wall and prismatic joints get
        # frozen. Back walls (x=±FIELD_X) are perpendicular to the rod axis
        # and rods don't reach them, so they can be full-height.
        SIDE_WALL_TOP = 0.825      # 9 mm below rod bottom
        WALL_HALF_H = 0.075        # full-height (back walls + caps)
        GOAL_HALF_W = self.cfg.goal_mouth_half_width
        GOAL_H = self.cfg.goal_height
        WALL_T = 0.01

        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.3, dynamic_friction=0.25, restitution=0.4
        )
        coll = sim_utils.CollisionPropertiesCfg()
        if self.cfg.visualize_primitive_collision:
            visual = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.95, 0.2))
        else:
            visual = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), opacity=0.0
            )

        def cuboid(name: str, size: tuple[float, float, float], pos: tuple[float, float, float]):
            cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=coll,
                physics_material=material,
                visual_material=visual,
            )
            cfg.func(f"{base}/{name}", cfg, translation=pos)

        cuboid("floor",
               size=(2 * FIELD_X, 2 * FIELD_Y, WALL_T),
               pos=(0.0, 0.0, FIELD_Z - WALL_T / 2))
        side_h = SIDE_WALL_TOP - FIELD_Z
        for sign in (+1, -1):
            cuboid(f"side_y_{'pos' if sign > 0 else 'neg'}",
                   size=(2 * FIELD_X, WALL_T, side_h),
                   pos=(0.0, sign * (FIELD_Y + WALL_T / 2), FIELD_Z + side_h / 2))
        for sign_y, suffix in ((+1, "y_pos"), (-1, "y_neg")):
            half_y_extent = (FIELD_Y - GOAL_HALF_W) / 2
            y_center = sign_y * (GOAL_HALF_W + half_y_extent)
            cuboid(f"back_x_pos_{suffix}",
                   size=(WALL_T, 2 * half_y_extent, 2 * WALL_HALF_H),
                   pos=(FIELD_X + WALL_T / 2, y_center, FIELD_Z + WALL_HALF_H))
            cuboid(f"back_x_neg_{suffix}",
                   size=(WALL_T, 2 * half_y_extent, 2 * WALL_HALF_H),
                   pos=(-(FIELD_X + WALL_T / 2), y_center, FIELD_Z + WALL_HALF_H))

        # Goal-mouth roofs: cap the gap above each goal opening so the ball
        # can't fly out the top. Spans the goal width in y; sits between the
        # top of the goal opening (FIELD_Z + GOAL_H) and the wall top
        # (FIELD_Z + 2 * WALL_HALF_H).
        roof_h = 2 * WALL_HALF_H - GOAL_H
        roof_z_center = FIELD_Z + GOAL_H + roof_h / 2
        for sign_x, name in ((+1, "roof_x_pos"), (-1, "roof_x_neg")):
            cuboid(name,
                   size=(WALL_T, 2 * GOAL_HALF_W, roof_h),
                   pos=(sign_x * (FIELD_X + WALL_T / 2), 0.0, roof_z_center))

        print(
            f">>> [foos_env] 9 primitive cuboid colliders spawned under {base}",
            flush=True,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        clipped = actions.clamp(-1.0, 1.0)
        if not self.cfg.self_play_mode:
            # Single-agent: one policy controls all 16 joints. Actions
            # directly map onto the full articulation.
            target_raw = self._joint_default + clipped * self._joint_scale
            self._prev_actions = self._last_actions
            self._last_actions = clipped
        else:
            # Self-play: `actions` is trainee's (N, 8) for team 1 only. The
            # opponent (frozen policy) emits team 2's 8 actions.
            trainee = clipped
            if self._opponent_actor is not None:
                opp_action = self._compute_opponent_action()
            else:
                opp_action = torch.zeros(self.num_envs, 8, device=self.device)
            # Trainee actions go to team-1 joint indices, opponent to team-2
            # indices. Order within each block already matches because both
            # teams share the same rod ordering.
            target_raw = self._joint_default.clone()
            target_raw[:, self._team1_idx] = trainee * self._joint_scale_team1
            target_raw[:, self._team2_idx] = opp_action * self._joint_scale_team2
            # Full-16 action history so smoothness diagnostics see both teams.
            full_actions = torch.zeros_like(self._action_targets)
            full_actions[:, self._team1_idx] = trainee
            full_actions[:, self._team2_idx] = opp_action
            self._prev_actions = self._last_actions
            self._last_actions = full_actions

        alpha = self.cfg.action_smoothing_alpha
        self._action_targets = alpha * target_raw + (1.0 - alpha) * self._action_targets

    def _compute_opponent_action(self) -> torch.Tensor:
        """Forward-pass through the frozen opponent using the previous
        step's observation. Returns (N, 8) clamped to [-1, 1]."""
        if self._last_observation is None:
            # First step after init/reset — opponent has no obs yet; hold
            # rods at default.
            return torch.zeros(self.num_envs, 8, device=self.device)
        return self._opponent_actor(self._last_observation)

    def _apply_action(self) -> None:
        self.table.set_joint_position_target(self._action_targets)

    def _get_observations(self) -> dict:
        joint_pos = self.table.data.joint_pos
        joint_vel = self.table.data.joint_vel

        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        ball_lin_vel = self.ball.data.root_lin_vel_w

        obs = torch.cat([joint_pos, joint_vel, ball_pos, ball_lin_vel], dim=-1)
        # Cache for the opponent forward pass on the *next* _pre_physics_step.
        self._last_observation = obs
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
        on_field_z = (bz > self.cfg.goal_z_min) & (bz < self.cfg.goal_z_max)
        past_x_neg = bx < -self.cfg.ball_x_limit
        past_x_pos = bx > self.cfg.ball_x_limit
        self._goal_team1 = past_x_neg & in_mouth & on_field_z   # team 2's net
        self._goal_team2 = past_x_pos & in_mouth & on_field_z   # team 1's net
        self._goal_any = self._goal_team1 | self._goal_team2
        off_mouth_x = (past_x_neg | past_x_pos) & ~(in_mouth & on_field_z)
        self._oob = (
            (by.abs() > self.cfg.ball_y_limit)
            | (bz < self.cfg.ball_z_min)
            | off_mouth_x
        )

        # Stuck-ball detection: planar speed below threshold for N straight
        # steps => dead ball. Counter resets to 0 the moment the ball moves,
        # and is zeroed on episode reset in _reset_idx. A ball that's already
        # OOB/goal this step is being reset anyway, so exclude it from the
        # stuck set to keep the accounting clean.
        if self._stuck_steps > 0:
            planar_speed = self.ball.data.root_lin_vel_w[:, :2].norm(dim=-1)
            slow = planar_speed < self.cfg.stuck_speed_threshold
            self._stuck_counter = torch.where(
                slow,
                self._stuck_counter + 1,
                torch.zeros_like(self._stuck_counter),
            )
            self._stuck = (self._stuck_counter >= self._stuck_steps) & ~(
                self._goal_any | self._oob
            )
        else:
            self._stuck = torch.zeros_like(self._oob)
        self._stuck_total += self._stuck.long()

        # Diagnostic prints (opt-in): heartbeat with ball state, plus
        # per-event prints for any termination this step.
        if self.cfg.debug_heartbeat:
            if not hasattr(self, "_diag_step"):
                self._diag_step = 0
            self._diag_step += 1
            # Dense heartbeat for the first 50 steps after each reset (to
            # capture what happens to a freshly-spawned ball), then sparse.
            ep_len = self.episode_length_buf[0].item()
            if (ep_len < 50 and ep_len % 2 == 0) or self._diag_step % 60 == 1:
                print(
                    f"[HB step={self._diag_step} ep_len={ep_len}] "
                    f"bx={bx[0].item():+.3f} by={by[0].item():+.3f} bz={bz[0].item():+.3f}  "
                    f"goal={int(self._goal_any[0])}  oob={int(self._oob[0])}  "
                    f"stuck={int(self._stuck[0])}({int(self._stuck_counter[0])}) "
                    f"in_mouth={int(in_mouth[0])} on_field_z={int(on_field_z[0])} "
                    f"past_x={int(past_x_neg[0]) - int(past_x_pos[0])}",
                    flush=True,
                )
        if getattr(self.cfg, "log_terminations", False):
            for env_id in range(self.num_envs):
                if self._goal_any[env_id]:
                    # _goal_team1 fires when the ball crossed into team 2's
                    # net (bx < -limit), meaning team 1 scored. Vice versa.
                    scorer = "team1" if self._goal_team1[env_id] else "team2"
                    net = "team2's net" if self._goal_team1[env_id] else "team1's net"
                    print(
                        f"[GOAL] scorer={scorer} into {net} env={env_id} "
                        f"bx={bx[env_id].item():+.3f} "
                        f"by={by[env_id].item():+.3f} bz={bz[env_id].item():+.3f}",
                        flush=True,
                    )
                elif self._oob[env_id]:
                    why = (
                        "off_mouth_x" if off_mouth_x[env_id] else
                        "y_oob" if by[env_id].abs() > self.cfg.ball_y_limit else
                        "z_below"
                    )
                    print(
                        f"[OOB:{why}] env={env_id} bx={bx[env_id].item():+.3f} "
                        f"by={by[env_id].item():+.3f} bz={bz[env_id].item():+.3f}",
                        flush=True,
                    )
                elif self._stuck[env_id]:
                    print(
                        f"[STUCK] env={env_id} dead ball after "
                        f"{self.cfg.stuck_time_s:.1f}s bx={bx[env_id].item():+.3f} "
                        f"by={by[env_id].item():+.3f} bz={bz[env_id].item():+.3f}",
                        flush=True,
                    )

        terminated = self._goal_any | self._oob
        # A stuck ball ends the point via truncation (time-limit-like: no
        # goal, value bootstraps) rather than a hard terminal.
        truncated = truncated | self._stuck

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

        if cfg.self_play_mode:
            # Trainee attacks toward -X (team 2's net at x=-0.6). Reward
            # leftward ball velocity directly. Potential-based: telescopes
            # to (x_init - x_final), so oscillating ball nets zero.
            v_toward = -ball_lin_vel[:, 0]
            # Zero-sum on goals: +1 when trainee scored, -1 when scored on.
            goal_reward = self._goal_team1.float() - self._goal_team2.float()
        else:
            # Single-agent: one policy controls all rods. Reward pushing
            # the ball toward *whichever* goal is closer (nearest-goal
            # shaping) and score in either net.
            sign_to_nearest = torch.sign(ball_pos_local[:, 0])
            v_toward = ball_lin_vel[:, 0] * sign_to_nearest
            goal_reward = self._goal_any.float()

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
            + cfg.rew_scale_goal * goal_reward
            + cfg.rew_scale_oob * self._oob.float()
        )

        self.extras["log"] = {
            "rew/ball_speed": (cfg.rew_scale_ball_speed * ball_speed).mean(),
            "rew/ball_x_speed": (cfg.rew_scale_ball_x_speed * ball_x_speed).mean(),
            "rew/v_toward_goal": (cfg.rew_scale_v_toward_goal * v_toward).mean(),
            "rew/action_cost": (cfg.rew_scale_action * action_cost).mean(),
            "rew/action_delta": (cfg.rew_scale_action_delta * action_delta).mean(),
            "rew/step": step_penalty.mean(),
            "rew/goal": (cfg.rew_scale_goal * goal_reward).mean(),
            "rew/oob": (cfg.rew_scale_oob * self._oob.float()).mean(),
            "score/goals_total": (self._score_team1 + self._score_team2).float().mean(),
            "score/team1_total": self._score_team1.float().mean(),
            "score/team2_total": self._score_team2.float().mean(),
            "score/stuck_total": self._stuck_total.float().mean(),
            "curriculum/progress": torch.tensor(
                min(
                    self.common_step_counter / max(1, cfg.curriculum_decay_steps),
                    1.0,
                ),
                device=self.device,
            ),
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

        # Curriculum spawn: ball positioned past the goalie + initial velocity
        # toward the same goal. Y jitter kept small so the ball lands on the
        # playing surface rather than on a side wall. Interpolate between the
        # easy bootstrap (`ball_spawn_*`) and the hard target
        # (`curriculum_target_*`) by the fraction of `curriculum_decay_steps`
        # consumed so far. self.common_step_counter is rl_games's frame
        # counter (ticks once per env-step across all envs).
        n = len(env_ids)
        progress = min(
            self.common_step_counter / max(1, self.cfg.curriculum_decay_steps), 1.0
        )
        near = (
            (1.0 - progress) * self.cfg.ball_spawn_x_near
            + progress * self.cfg.curriculum_target_x_near
        )
        far = (
            (1.0 - progress) * self.cfg.ball_spawn_x_far
            + progress * self.cfg.curriculum_target_x_far
        )
        vx = (
            (1.0 - progress) * self.cfg.ball_spawn_vx_toward_goal
            + progress * self.cfg.curriculum_target_vx
        )
        if near == 0.0 and far == 0.0:
            x_offset = torch.zeros(n, 1, device=self.device)
            # Always launch toward +X goal when spawn is fixed at center.
            sign = torch.ones(n, 1, device=self.device)
        else:
            mag = torch.empty(n, 1, device=self.device).uniform_(near, far)
            sign = torch.where(
                torch.rand(n, 1, device=self.device) < 0.5,
                torch.tensor(-1.0, device=self.device),
                torch.tensor(1.0, device=self.device),
            )
            x_offset = sign * mag
        y_offset = torch.empty(n, 1, device=self.device).uniform_(-0.05, 0.05)
        ball_root_state = self.ball.data.default_root_state[env_ids].clone()
        ball_root_state[:, :3] += self.scene.env_origins[env_ids]
        ball_root_state[:, 0:1] += x_offset
        ball_root_state[:, 1:2] += y_offset
        ball_root_state[:, 7:13] = 0.0  # zero linear + angular velocity
        # Pre-launch ball toward closer goal (sign already encodes direction).
        ball_root_state[:, 7:8] = sign * vx
        # Optional extra random kick on top of the directed push.
        vmax = self.cfg.ball_spawn_random_vmax
        if vmax > 0:
            ball_root_state[:, 7:8] += torch.empty(
                n, 1, device=self.device
            ).uniform_(-vmax, vmax)
            ball_root_state[:, 8:9] = torch.empty(
                n, 1, device=self.device
            ).uniform_(-vmax, vmax)
        self.ball.write_root_pose_to_sim(ball_root_state[:, :7], env_ids=env_ids)
        self.ball.write_root_velocity_to_sim(ball_root_state[:, 7:], env_ids=env_ids)

        self._last_actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        # Re-anchor the low-pass-filtered targets at the neutral pose so the
        # next episode doesn't inherit the previous one's smoothed trajectory.
        self._action_targets[env_ids] = self._joint_default[env_ids]
        # Fresh serve => clear the stuck *counter* so the newly-placed (and
        # possibly momentarily-stationary) ball doesn't inherit the previous
        # point's stuck streak. Deliberately do NOT clear self._stuck here:
        # it's fully recomputed each _get_dones, and clearing it would wipe
        # this step's flag before an external reader (the match runner, which
        # reads inner._stuck after env.step() returns) can see it — _reset_idx
        # runs inside step(), after _get_dones.
