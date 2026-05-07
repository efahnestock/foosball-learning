import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# All numeric values below are TUNABLE first-pass guesses. Real foosball balls
# are roughly 17 mm radius and 25 g; friction and restitution are eyeballed
# and will need to be verified once the viewer is running.
BALL_CFG = RigidObjectCfg(
    spawn=sim_utils.SphereCfg(
        radius=0.017,  # TUNABLE
        mass_props=sim_utils.MassPropertiesCfg(mass=0.025),  # TUNABLE
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            linear_damping=0.05,
            angular_damping=0.05,
            max_depenetration_velocity=2.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.3,   # TUNABLE
            dynamic_friction=0.25, # TUNABLE
            restitution=0.6,       # TUNABLE
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.95, 0.2),
        ),
    ),
    # Drop above the playing surface. After the table is lifted by 0.482m,
    # the playing field sits around z ~= 0.84; spawning at z=1.0 drops the
    # ball ~16 cm onto the field.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
)
