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
            # Bright orange — stands out against the wood-toned table and
            # both rod-team colors.
            diffuse_color=(1.0, 0.4, 0.0),
        ),
    ),
    # Spawn the ball INSIDE the playing-field cavity. Field surface is at
    # z=0.762, ball-at-rest center ≈ 0.779. Spawn ~1 cm above surface so it
    # drops cleanly without phasing through the cuboid floor.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.79)),
)
