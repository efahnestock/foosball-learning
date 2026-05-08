from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
_URDF_PATH = _REPO_ROOT / "assets" / "foosball_table" / "foosball_table.urdf"

# All numeric values below are TUNABLE first-pass guesses. The URDF inertias
# (mass=0.7638 with off-diagonal zeros on every rod) are also suspect and
# likely need revisiting once the viewer shows real rod behavior.
FOOSBALL_TABLE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(_URDF_PATH),
        fix_base=True,
        # Load-bearing: the URDF uses `empty_*` shell links between the table
        # and each rod to chain the prismatic + revolute joints. Merging fixed
        # joints would collapse that chain.
        merge_fixed_joints=False,
        # Decompose the table mesh into multiple convex pieces so the ball
        # can settle inside the playing-field cavity. Default `convex_hull`
        # turns the whole table into a single closed box and the ball floats
        # on top of it.
        collider_type="convex_decomposition",
        # PD gains baked into the converted USD. Runtime ImplicitActuatorCfg
        # below overrides them; these only need to be sensible enough for the
        # USD itself to load. Match the actuator values for consistency.
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    ".*_prismatic_joint": 800.0,
                    ".*_revolute_joint": 20.0,
                },
                damping={
                    ".*_prismatic_joint": 40.0,
                    ".*_revolute_joint": 1.0,
                },
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Lift the table so the leg bottoms sit on the ground plane. The
        # mesh's bottom-of-legs is at table-frame z = -0.482, so a +0.482
        # offset puts them at world z = 0.
        pos=(0.0, 0.0, 0.482),
        joint_pos={
            ".*_prismatic_joint": 0.0,
            ".*_revolute_joint": 0.0,
        },
    ),
    actuators={
        "prismatic": ImplicitActuatorCfg(
            joint_names_expr=[".*_prismatic_joint"],
            stiffness=800.0,        # TUNABLE
            damping=40.0,           # TUNABLE
            effort_limit_sim=200.0, # TUNABLE
            velocity_limit_sim=5.0, # TUNABLE
        ),
        "revolute": ImplicitActuatorCfg(
            joint_names_expr=[".*_revolute_joint"],
            stiffness=20.0,          # TUNABLE
            damping=1.0,             # TUNABLE — keeps rods responsive enough
                                     # to snap-kick. Tried 5.0 to reduce twitch
                                     # but it stalled training; the action_delta
                                     # reward penalty handles smoothness instead.
            effort_limit_sim=20.0,   # TUNABLE
            velocity_limit_sim=30.0, # TUNABLE — high to allow snap-kicks. 12.0
                                     # made kicking infeasible.
        ),
    },
)
