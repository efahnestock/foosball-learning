"""Generate the foosball table collision asset.

Produces a single composite OBJ + USD with convex_decomposition collision props.
Spawning one USD per env is cheaper than 7 separate primitive cuboid colliders
(saves ~28k extra prims at num_envs=4096).

Run from the repo root:

    OMNI_KIT_ACCEPT_EULA=YES python scripts/build_table_collision.py

Output:
    assets/foosball_table/meshes/collision/table_collision.obj
    assets/foosball_table/meshes/collision/table_collision.usd  (+ Props/)
"""

from __future__ import annotations

import argparse
from pathlib import Path

# AppLauncher must come first if Isaac Sim is needed for the USD step.
# trimesh-only OBJ generation works without it; do that part standalone.

OUT_DIR = Path("assets/foosball_table/meshes/collision")

# Field constants — must match FoosEnvCfg defaults.
FIELD_X = 0.705
FIELD_Y = 0.39
FIELD_Z = 0.79
WALL_HALF_H = 0.07
GOAL_HALF_W = 0.10
WALL_T = 0.01


def build_obj() -> Path:
    import trimesh
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = OUT_DIR / "table_collision.obj"

    boxes: list[trimesh.Trimesh] = []

    def box(size: tuple[float, float, float], pos: tuple[float, float, float]):
        m = trimesh.creation.box(extents=size)
        m.apply_translation(pos)
        boxes.append(m)

    # Floor
    box((2 * FIELD_X, 2 * FIELD_Y, WALL_T), (0, 0, FIELD_Z - WALL_T / 2))
    # Side walls along the long edges
    for sign in (+1, -1):
        box((2 * FIELD_X, WALL_T, 2 * WALL_HALF_H),
            (0, sign * (FIELD_Y + WALL_T / 2), FIELD_Z + WALL_HALF_H))
    # Back walls flanking each goal mouth
    half_y_extent = (FIELD_Y - GOAL_HALF_W) / 2
    for sx in (+1, -1):
        for sy in (+1, -1):
            y_center = sy * (GOAL_HALF_W + half_y_extent)
            box((WALL_T, 2 * half_y_extent, 2 * WALL_HALF_H),
                (sx * (FIELD_X + WALL_T / 2), y_center, FIELD_Z + WALL_HALF_H))

    composite = trimesh.util.concatenate(boxes)
    obj_path.write_text(trimesh.exchange.obj.export_obj(
        composite, include_normals=True, include_color=False, include_texture=False
    ))
    print(f"wrote OBJ ({len(composite.vertices)} v / {len(composite.faces)} f) -> {obj_path}")
    return obj_path


def build_usd(obj_path: Path) -> None:
    """Convert the OBJ to USD with convex_decomposition collision props."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    app_launcher = AppLauncher(args)
    app = app_launcher.app

    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg

    cfg = MeshConverterCfg(
        asset_path=str(obj_path.resolve()),
        usd_dir=str(OUT_DIR.resolve()),
        usd_file_name="table_collision.usd",
        force_usd_conversion=True,
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        # 7 boxes -> at most ~14 hulls (some may merge). Plenty.
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=16,
            hull_vertex_limit=64,
            voxel_resolution=500_000,
            error_percentage=2.0,
            shrink_wrap=False,
        ),
    )
    conv = MeshConverter(cfg)
    print(f"wrote USD -> {conv.usd_path}")
    app.close()


def main() -> None:
    obj = build_obj()
    build_usd(obj)


if __name__ == "__main__":
    main()
