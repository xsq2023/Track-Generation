#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from PIL import Image
from pygltflib import GLTF2, Node

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from path_defaults import DEFAULT_OUTPUT_ROOT, DEFAULT_SOURCE_ROOT

STEP0_DIRNAME = "step0"
SUMMARY_PATH_NAME = "_batch_summary.tsv"
NAVMESH_CACHE_DIRNAME = "navmesh_cache"
LOG_DIRNAME = "_batch_logs"

SUMMARY_FIELDS = [
    "row_id",
    "idx",
    "total",
    "scene_id",
    "scene_path",
    "status",
    "run_state",
    "quarantine",
    "elapsed_sec",
    "scene_init_ok",
    "navmesh_status",
    "fallback_plan",
    "worker_exit_code",
    "fail_reason",
    "scene_init_json",
    "bootstrap_json",
    "log_path",
]


def scene_id_from_path(scene_path: Path) -> str:
    raw = scene_path.stem
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw).strip("_")
    return cleaned or "unknown_scene"


def to_json_compatible(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=to_json_compatible), encoding="utf-8")


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def vector_to_list(vec) -> List[float]:
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def aabb_to_dict(bb) -> Dict:
    min_v = np.array(vector_to_list(bb.min), dtype=np.float64)
    max_v = np.array(vector_to_list(bb.max), dtype=np.float64)
    size = max_v - min_v
    center = (min_v + max_v) * 0.5
    return {
        "center": center.tolist(),
        "size": size.tolist(),
        "min": min_v.tolist(),
        "max": max_v.tolist(),
        "max_dim": float(np.max(size)),
        "volume": float(np.prod(size)),
    }


def collect_env_meta() -> Dict:
    subset = []
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        keys = ("habitat", "magnum", "corrade", "numpy", "opencv")
        for line in (proc.stdout or "").splitlines():
            low = line.lower()
            if any(k in low for k in keys):
                subset.append(line.strip())
    except Exception:
        subset = []

    versions = {}
    try:
        versions["habitat_sim"] = getattr(habitat_sim, "__version__", None)
    except Exception:
        versions["habitat_sim"] = None
    try:
        versions["numpy"] = np.__version__
    except Exception:
        versions["numpy"] = None
    try:
        versions["magnum"] = getattr(mn, "__version__", None)
    except Exception:
        versions["magnum"] = None

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "pid": os.getpid(),
        "pip_freeze_subset": subset,
        "module_versions": versions,
    }


def build_sim(
    scene_path: Path,
    width: int,
    height: int,
    hfov: float,
    zfar: float,
    seed: int,
    with_sensors: bool,
) -> habitat_sim.Simulator:
    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": str(scene_path),
            "width": int(width),
            "height": int(height),
            "hfov": float(hfov),
            "zfar": float(zfar),
            "sensor_height": 1.5,
            "seed": int(seed),
            "silent": True,
            "enable_physics": False,
            "default_agent_navmesh": False,
            "color_sensor": bool(with_sensors),
            "depth_sensor": bool(with_sensors),
            "semantic_sensor": False,
            "frustum_culling": True,
        }
    )
    cfg = make_cfg(settings)
    sim = habitat_sim.Simulator(cfg)
    sim.seed(seed)
    return sim


def compute_scene_aabb(sim: habitat_sim.Simulator) -> Dict:
    return aabb_to_dict(sim.scene_aabb)


def compute_transform_plan(
    aabb: Dict,
    target_min_dim: float,
    target_max_dim: float,
    align_floor_to_zero: bool,
) -> Dict:
    max_dim = float(aabb["max_dim"])
    center = np.asarray(aabb["center"], dtype=np.float64)
    min_v = np.asarray(aabb["min"], dtype=np.float64)
    scale = 1.0
    if max_dim > 1e-9:
        if max_dim < target_min_dim:
            scale = float(target_min_dim / max_dim)
        elif max_dim > target_max_dim:
            scale = float(target_max_dim / max_dim)
    translation = np.array(
        [-center[0] * scale, -center[1] * scale, -center[2] * scale],
        dtype=np.float64,
    )
    if align_floor_to_zero:
        translation[1] = -min_v[1] * scale
    return {
        "scale": float(scale),
        "translation": translation.tolist(),
        "align_floor_to_zero": bool(align_floor_to_zero),
        "target_max_dim": float(target_max_dim),
        "target_min_dim": float(target_min_dim),
        "requested": True,
        "applied": False,
        "apply_mode": "scene_instance_uniform_scale_translation",
    }


def apply_transform_to_aabb(aabb: Dict, scale: float, translation: List[float]) -> Dict:
    min_v = np.asarray(aabb["min"], dtype=np.float64) * scale + np.asarray(translation, dtype=np.float64)
    max_v = np.asarray(aabb["max"], dtype=np.float64) * scale + np.asarray(translation, dtype=np.float64)
    size = max_v - min_v
    center = (min_v + max_v) * 0.5
    return {
        "center": center.tolist(),
        "size": size.tolist(),
        "min": min_v.tolist(),
        "max": max_v.tolist(),
        "max_dim": float(np.max(size)),
        "volume": float(np.prod(size)),
    }


def write_transformed_scene_glb(scene_path: Path, scene_dir: Path, transform_plan: Dict) -> Path:
    out_path = scene_dir / "scene_post_transform.glb"
    scale = float(transform_plan["scale"])
    world_translation = [float(v) for v in transform_plan["translation"]]
    # Habitat's GLB loader maps glTF root-node translation axes to world as:
    # x->x, y->-z, z->y. Convert the desired world translation back to glTF space.
    translation = [
        world_translation[0],
        -world_translation[2],
        world_translation[1],
    ]

    gltf = GLTF2().load_binary(str(scene_path))
    scene_index = gltf.scene if gltf.scene is not None else 0
    if scene_index >= len(gltf.scenes):
        raise RuntimeError(f"gltf_scene_index_invalid:{scene_index}")

    scene = gltf.scenes[scene_index]
    old_roots = list(scene.nodes or [])
    if not old_roots:
        raise RuntimeError("gltf_scene_has_no_root_nodes")

    new_root = Node(
        name="trackgen_root_transform",
        children=old_roots,
        scale=[scale, scale, scale],
        translation=translation,
    )
    gltf.nodes.append(new_root)
    scene.nodes = [len(gltf.nodes) - 1]
    gltf.save_binary(str(out_path))
    return out_path


def aabb_diff_summary(expected: Optional[Dict], actual: Optional[Dict]) -> Dict:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return {
            "matches": False,
            "atol": None,
            "max_abs_err": None,
            "per_key_max_abs_err": {},
        }

    per_key = {}
    max_abs_err = 0.0
    for key in ("center", "size", "min", "max"):
        exp_v = np.asarray(expected.get(key, []), dtype=np.float64)
        act_v = np.asarray(actual.get(key, []), dtype=np.float64)
        if exp_v.shape != act_v.shape or exp_v.size == 0:
            per_key[key] = None
            max_abs_err = math.inf
            continue
        err = float(np.max(np.abs(exp_v - act_v)))
        per_key[key] = err
        max_abs_err = max(max_abs_err, err)

    atol = 1.0e-3
    return {
        "matches": bool(np.isfinite(max_abs_err) and max_abs_err <= atol),
        "atol": float(atol),
        "max_abs_err": None if not np.isfinite(max_abs_err) else float(max_abs_err),
        "per_key_max_abs_err": per_key,
        "expected_max_dim": expected.get("max_dim"),
        "actual_max_dim": actual.get("max_dim"),
    }


def build_debug_cam_pose(aabb: Dict, fov_deg: float, near: float, far: float) -> Dict:
    center = np.asarray(aabb["center"], dtype=np.float64)
    max_dim = max(float(aabb["max_dim"]), 1.0)
    distance_raw = max_dim * 1.2
    distance = float(np.clip(distance_raw, 4.0, 50.0))
    height = max(1.5, distance * 0.35)

    position = center + np.array([0.0, height, distance], dtype=np.float64)
    target = center.copy()
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        forward = np.array([0.0, -0.3, -0.95], dtype=np.float64)
        norm = np.linalg.norm(forward)
    forward = forward / norm

    view = mn.Matrix4.look_at(
        mn.Vector3(position.tolist()),
        mn.Vector3(target.tolist()),
        mn.Vector3(up.tolist()),
    )
    q = mn.Quaternion.from_matrix(view.rotation())
    yaw = float(math.atan2(forward[0], -forward[2]))
    pitch = float(math.asin(np.clip(forward[1], -1.0, 1.0)))

    return {
        "position": position.tolist(),
        "look_at_target": target.tolist(),
        "up": up.tolist(),
        "fov_deg": float(fov_deg),
        "near": float(near),
        "far": float(far),
        "debug_cam_distance_m": distance,
        "debug_cam_distance_raw_m": float(distance_raw),
        "rotation_quat_wxyz": [float(q.scalar), float(q.vector.x), float(q.vector.y), float(q.vector.z)],
        "forward": forward.tolist(),
        "yaw_rad": yaw,
        "pitch_rad": pitch,
        "roll_rad": 0.0,
        "pose_frame": "world",
    }


def apply_debug_camera_pose(sim: habitat_sim.Simulator, pose: Dict) -> Dict:
    agent = sim.initialize_agent(0)
    pos = np.asarray(pose["position"], dtype=np.float32)
    target = np.asarray(pose["look_at_target"], dtype=np.float32)
    up = np.asarray(pose["up"], dtype=np.float32)

    view = mn.Matrix4.look_at(
        mn.Vector3(pos.tolist()),
        mn.Vector3(target.tolist()),
        mn.Vector3(up.tolist()),
    )
    q = mn.Quaternion.from_matrix(view.rotation())
    coeff = np.array([q.vector.x, q.vector.y, q.vector.z, q.scalar], dtype=np.float32)

    state = agent.get_state()
    state.position = pos
    state.rotation = coeff
    agent.set_state(state)
    rb = agent.get_state()
    return {
        "position": [float(rb.position[0]), float(rb.position[1]), float(rb.position[2])],
        "rotation_quat_wxyz": [
            float(rb.rotation.real),
            float(rb.rotation.imag[0]),
            float(rb.rotation.imag[1]),
            float(rb.rotation.imag[2]),
        ],
    }


def save_depth_png_u16(path: Path, depth_u16: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_u16).save(path)


def render_debug_observations(
    sim: habitat_sim.Simulator,
    pose: Dict,
    scene_dir: Path,
    tag: str,
    depth_clip_max_m: float,
    depth_auto_percentile: float,
) -> Tuple[Dict, Dict]:
    readback = apply_debug_camera_pose(sim, pose)
    obs = sim.get_sensor_observations()

    rgb = np.asarray(obs["color_sensor"])
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = np.asarray(rgb, dtype=np.uint8)

    depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
    depth_finite = np.isfinite(depth)
    depth_valid_mask = depth_finite & (depth > 0.0)
    valid = depth[depth_valid_mask]
    valid_ratio = float(np.mean(depth_valid_mask))

    depth_safe = np.where(depth_finite, depth, 0.0)
    depth_clip = np.clip(depth_safe, 0.0, depth_clip_max_m)
    depth_clip_img = np.round((depth_clip / max(depth_clip_max_m, 1e-6)) * 255.0).astype(np.uint8)
    depth_u16_clip = np.round((depth_clip / max(depth_clip_max_m, 1e-6)) * 65535.0).astype(np.uint16)

    if valid.size > 0:
        auto_max = float(max(np.percentile(valid, depth_auto_percentile), 1e-6))
        auto_min = float(max(np.percentile(valid, 1.0), 0.0))
    else:
        auto_max = float(depth_clip_max_m)
        auto_min = 0.0
    auto_den = max(auto_max - auto_min, 1e-6)
    depth_auto01 = np.clip((depth_safe - auto_min) / auto_den, 0.0, 1.0)
    depth_vis = np.round(depth_auto01 * 255.0).astype(np.uint8)
    depth_u16_auto = np.round(np.clip(depth_safe / max(auto_max, 1e-6), 0.0, 1.0) * 65535.0).astype(np.uint16)

    rgb_path = scene_dir / f"debug_rgb_{tag}.png"
    depth_path = scene_dir / f"debug_depth_{tag}.png"
    depth_clip_path = scene_dir / f"debug_depth_clip_{tag}.png"
    depth_npy_path = scene_dir / f"debug_depth_{tag}.npy"
    depth_u16_path = scene_dir / f"debug_depth_u16_{tag}.png"
    depth_u16_clip_path = scene_dir / f"debug_depth_u16_clip_{tag}.png"
    depth_u16_p99_path = scene_dir / f"debug_depth_u16_p99_{tag}.png"

    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(depth_vis).save(depth_path)
    Image.fromarray(depth_clip_img).save(depth_clip_path)
    np.save(depth_npy_path, depth)
    save_depth_png_u16(depth_u16_path, depth_u16_auto)
    save_depth_png_u16(depth_u16_clip_path, depth_u16_clip)
    save_depth_png_u16(depth_u16_p99_path, depth_u16_auto)

    meta = {
        "valid_ratio": valid_ratio,
        "valid_min_m": float(np.min(valid)) if valid.size > 0 else None,
        "valid_max_m": float(np.max(valid)) if valid.size > 0 else None,
        "valid_mean_m": float(np.mean(valid)) if valid.size > 0 else None,
        "clip_max_m": float(depth_clip_max_m),
        "rgb_mean": float(np.mean(rgb)),
        "rgb_std": float(np.std(rgb)),
        "rgb_min": int(np.min(rgb)),
        "rgb_max": int(np.max(rgb)),
        "depth_range_valid_min_m": float(np.min(valid)) if valid.size > 0 else None,
        "depth_range_valid_max_m": float(np.max(valid)) if valid.size > 0 else None,
        "depth_range_valid_mean_m": float(np.mean(valid)) if valid.size > 0 else None,
        "u16_clip_max_m": float(depth_clip_max_m),
        "u16_auto_max_m": float(auto_max),
        "u16_auto_percentile": float(depth_auto_percentile),
        "u16_saturation_ratio_clip": float(np.mean(depth_valid_mask & (depth_safe >= depth_clip_max_m)))
        if valid.size > 0
        else None,
        "u16_saturation_ratio_auto": float(np.mean(depth_valid_mask & (depth_safe >= auto_max)))
        if valid.size > 0
        else None,
        "u16_saturation_ratio": float(np.mean(depth_valid_mask & (depth_safe >= auto_max))) if valid.size > 0 else None,
        "vis_percentile_min_m": float(auto_min) if valid.size > 0 else None,
        "vis_percentile_max_m": float(auto_max) if valid.size > 0 else None,
        "clip_fixed_max_m": float(depth_clip_max_m),
        "artifacts": {
            "debug_rgb": str(rgb_path),
            "debug_depth": str(depth_path),
            "debug_depth_clip": str(depth_clip_path),
            "debug_depth_npy": str(depth_npy_path),
            "debug_depth_u16": str(depth_u16_path),
            "debug_depth_u16_clip": str(depth_u16_clip_path),
            "debug_depth_u16_p99": str(depth_u16_p99_path),
        },
    }
    return meta, readback


def navmesh_poly_count(pathfinder) -> Optional[int]:
    try:
        idx = pathfinder.build_navmesh_vertex_indices()
        if idx is None:
            return None
        return int(len(idx) // 3)
    except Exception:
        return None


def validate_navmesh(pathfinder) -> Tuple[bool, Optional[str]]:
    if not bool(pathfinder.is_loaded):
        return False, "pathfinder_not_loaded"
    try:
        p = pathfinder.get_random_navigable_point()
    except Exception as exc:
        return False, f"random_navigable_exception:{exc}"
    arr = np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False, "random_navigable_nonfinite"
    return True, None


def nav_settings_summary(nav, wall_ms: float, poly_count: Optional[int]) -> Dict:
    return {
        "agent_radius": float(nav.agent_radius),
        "agent_height": float(nav.agent_height),
        "cell_size": float(nav.cell_size),
        "cell_height": float(nav.cell_height),
        "agent_max_climb": float(nav.agent_max_climb),
        "agent_max_slope": float(nav.agent_max_slope),
        "attempt_wall_time_ms": float(wall_ms),
        "navmesh_poly_count": poly_count,
    }


def make_navmesh_skip_result(reason: str) -> Dict:
    return {
        "navmesh_status": "FAIL",
        "navmesh_fail_reason": reason,
        "fallback_used": "no_navmesh_sampling",
        "fallback_plan": "no_navmesh_sampling",
        "navmesh_attempts": [
            {
                "attempt_id": "SKIP_POLICY",
                "settings_summary": None,
                "success": False,
                "error": reason,
                "attempt_wall_time_ms": 0.0,
                "navmesh_poly_count": None,
            }
        ],
        "navmesh_cache_key": None,
        "navmesh_cache_path": None,
    }


def run_navmesh_strategy(
    sim: habitat_sim.Simulator,
    scene_id: str,
    cache_dir: Path,
    max_dim: float,
    allow_navmesh: bool,
    skip_reason: Optional[str],
) -> Dict:
    if not allow_navmesh:
        return make_navmesh_skip_result(skip_reason or "navmesh_skipped")

    attempts = []
    cache_key_seed = f"{scene_id}|{sim.config.sim_cfg.scene_id}"
    cache_key = hashlib.sha1(cache_key_seed.encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"{scene_id}_{cache_key}.navmesh"
    cache_dir.mkdir(parents=True, exist_ok=True)

    status = "FAIL"
    fail_reason = None
    fallback_used = "no_navmesh_sampling"
    last_error = None
    success_attempt_id = None

    t0 = time.time()
    if cache_path.exists():
        try:
            loaded = bool(sim.pathfinder.load_nav_mesh(str(cache_path)))
            ok, validate_error = validate_navmesh(sim.pathfinder) if loaded else (False, "cache_load_false")
            success = bool(loaded and ok)
            error = validate_error
        except Exception as exc:
            success = False
            error = f"cache_load_exception:{exc}"
        attempts.append(
            {
                "attempt_id": "LOAD_CACHE",
                "settings_summary": None,
                "success": success,
                "error": error,
                "attempt_wall_time_ms": float((time.time() - t0) * 1000.0),
                "navmesh_poly_count": navmesh_poly_count(sim.pathfinder) if success else None,
            }
        )
        if success:
            status = "OK"
            fallback_used = "none"
            return {
                "navmesh_status": status,
                "navmesh_fail_reason": None,
                "fallback_used": fallback_used,
                "fallback_plan": "navmesh",
                "navmesh_attempts": attempts,
                "navmesh_cache_key": cache_key,
                "navmesh_cache_path": str(cache_path),
            }
        last_error = error
    else:
        attempts.append(
            {
                "attempt_id": "LOAD_CACHE",
                "settings_summary": None,
                "success": False,
                "error": "cache_missing",
                "attempt_wall_time_ms": float((time.time() - t0) * 1000.0),
                "navmesh_poly_count": None,
            }
        )
        last_error = "cache_missing"

    configs = []
    nav_a = habitat_sim.nav.NavMeshSettings()
    nav_a.set_defaults()
    configs.append(("ATTEMPT_A_DEFAULT", nav_a))

    nav_b = habitat_sim.nav.NavMeshSettings()
    nav_b.set_defaults()
    nav_b.cell_size *= 2.0
    nav_b.cell_height *= 2.0
    nav_b.agent_radius = max(nav_b.agent_radius * 0.8, 0.02)
    nav_b.agent_height = max(nav_b.agent_height * 0.85, 0.2)
    configs.append(("ATTEMPT_B_RELAXED", nav_b))

    nav_c = habitat_sim.nav.NavMeshSettings()
    nav_c.set_defaults()
    nav_c.cell_size *= 3.0
    nav_c.cell_height *= 3.0
    nav_c.agent_radius = max(nav_c.agent_radius * 0.6, 0.01)
    nav_c.agent_height = max(nav_c.agent_height * 0.7, 0.15)
    nav_c.agent_max_climb = max(nav_c.agent_max_climb * 2.0, nav_c.agent_max_climb + 0.2)
    nav_c.agent_max_slope = min(nav_c.agent_max_slope + 15.0, 85.0)
    configs.append(("ATTEMPT_C_RELAXED_MORE", nav_c))

    for attempt_id, nav in configs:
        t_start = time.time()
        success = False
        error = None
        poly_count = None
        try:
            recomputed = bool(sim.recompute_navmesh(sim.pathfinder, nav))
            if recomputed:
                ok, validate_error = validate_navmesh(sim.pathfinder)
                success = bool(ok)
                if not success:
                    error = validate_error
                else:
                    poly_count = navmesh_poly_count(sim.pathfinder)
            else:
                error = "recompute_navmesh_false"
        except Exception as exc:
            success = False
            error = f"recompute_navmesh_exception:{exc}"

        wall_ms = (time.time() - t_start) * 1000.0
        attempts.append(
            {
                "attempt_id": attempt_id,
                "settings_summary": nav_settings_summary(nav, wall_ms, poly_count),
                "success": success,
                "error": error,
                "attempt_wall_time_ms": float(wall_ms),
                "navmesh_poly_count": poly_count,
            }
        )
        if success:
            try:
                sim.pathfinder.save_nav_mesh(str(cache_path))
            except Exception:
                pass
            status = "RECOMPUTED_OK"
            success_attempt_id = attempt_id
            break
        last_error = error

    if status == "RECOMPUTED_OK":
        if success_attempt_id in {"ATTEMPT_B_RELAXED", "ATTEMPT_C_RELAXED_MORE"}:
            fallback_used = "relaxed_navmesh"
        else:
            fallback_used = "none"

    if status == "FAIL":
        if max_dim < 1.0:
            fail_reason = "scene_scale_too_small"
        elif max_dim > 200.0:
            fail_reason = "scene_scale_too_large"
        else:
            fail_reason = str(last_error or "navmesh_failed")

    return {
        "navmesh_status": status,
        "navmesh_fail_reason": fail_reason,
        "fallback_used": fallback_used,
        "fallback_plan": "navmesh" if status in {"OK", "RECOMPUTED_OK"} else "no_navmesh_sampling",
        "navmesh_attempts": attempts,
        "navmesh_cache_key": cache_key,
        "navmesh_cache_path": str(cache_path),
    }


def copy_post_artifacts_to_default(scene_dir: Path):
    pairs = [
        ("debug_rgb_post.png", "debug_rgb.png"),
        ("debug_depth_post.png", "debug_depth.png"),
        ("debug_depth_clip_post.png", "debug_depth_clip.png"),
        ("debug_depth_post.npy", "debug_depth.npy"),
        ("debug_depth_u16_post.png", "debug_depth_u16.png"),
        ("debug_depth_u16_clip_post.png", "debug_depth_u16_clip.png"),
        ("debug_depth_u16_p99_post.png", "debug_depth_u16_p99.png"),
    ]
    for src_name, dst_name in pairs:
        src = scene_dir / src_name
        if src.exists():
            shutil.copy2(src, scene_dir / dst_name)


def write_bootstrap(
    bootstrap_path: Path,
    scene_id: str,
    scene_path: Path,
    stage: str,
    detail: Optional[str] = None,
    extra: Optional[Dict] = None,
):
    payload = load_json(bootstrap_path) or {
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "pid": os.getpid(),
        "created_at_unix": time.time(),
        "events": [],
    }
    event = {
        "ts_unix": time.time(),
        "stage": stage,
        "detail": detail,
    }
    if extra:
        event["extra"] = extra
    payload["events"].append(event)
    payload["last_stage"] = stage
    payload["updated_at_unix"] = event["ts_unix"]
    save_json(bootstrap_path, payload)


def classify_anomaly(max_dim: Optional[float], volume: Optional[float], args) -> Dict:
    soft = False
    hard = False
    reasons = []
    if max_dim is not None:
        if max_dim < 1.0:
            soft = True
            reasons.append("max_dim_lt_1m")
        if max_dim > float(args.anomaly_max_dim_soft):
            soft = True
            reasons.append("max_dim_gt_soft_threshold")
        if max_dim > float(args.anomaly_max_dim_hard):
            hard = True
            reasons.append("max_dim_gt_hard_threshold")
    if volume is not None and volume > float(args.anomaly_aabb_volume_hard):
        hard = True
        reasons.append("aabb_volume_gt_hard_threshold")
    level = "hard" if hard else ("soft" if soft else "normal")
    return {
        "level": level,
        "soft": bool(soft),
        "hard": bool(hard),
        "reasons": reasons,
        "max_dim": max_dim,
        "aabb_volume": volume,
    }


def run_scene_stats_pass(scene_paths: List[Path], source_root: Path, stats_path: Path, seed: int, args) -> Dict:
    max_dims = []
    volumes = []
    entries = []
    anomaly_soft = []
    anomaly_hard = []

    for scene_path in scene_paths:
        scene_id = scene_id_from_path(scene_path)
        sim = None
        try:
            sim = build_sim(
                scene_path=scene_path,
                width=64,
                height=64,
                hfov=90.0,
                zfar=1000.0,
                seed=seed,
                with_sensors=False,
            )
            aabb = compute_scene_aabb(sim)
            max_dim = float(aabb["max_dim"])
            volume = float(aabb["volume"])
            anomaly = classify_anomaly(max_dim=max_dim, volume=volume, args=args)
            row = {
                "scene_id": scene_id,
                "scene_path": str(scene_path),
                "max_dim": max_dim,
                "aabb_volume": volume,
                "anomaly_level": anomaly["level"],
                "anomaly_reasons": anomaly["reasons"],
            }
            entries.append(row)
            max_dims.append(max_dim)
            volumes.append(volume)
            if anomaly["soft"]:
                anomaly_soft.append(scene_id)
            if anomaly["hard"]:
                anomaly_hard.append(scene_id)
        except Exception as exc:
            entries.append(
                {
                    "scene_id": scene_id,
                    "scene_path": str(scene_path),
                    "error": str(exc),
                    "error_code": "SCENE_STATS_PROBE_FAILED",
                }
            )
        finally:
            if sim is not None:
                try:
                    sim.close()
                except Exception:
                    pass

    stats = {
        "source_root": str(source_root),
        "scene_count": int(len(scene_paths)),
        "thresholds": {
            "anomaly_max_dim_soft": float(args.anomaly_max_dim_soft),
            "anomaly_max_dim_hard": float(args.anomaly_max_dim_hard),
            "anomaly_aabb_volume_hard": float(args.anomaly_aabb_volume_hard),
        },
        "max_dim": {
            "min": float(min(max_dims)) if max_dims else None,
            "median": float(np.median(max_dims)) if max_dims else None,
            "p95": percentile(max_dims, 95.0),
        },
        "aabb_volume": {
            "min": float(min(volumes)) if volumes else None,
            "median": float(np.median(volumes)) if volumes else None,
            "p95": percentile(volumes, 95.0),
        },
        "anomaly_scenes": sorted(set(anomaly_soft)),
        "anomaly_soft_scenes": sorted(set(anomaly_soft)),
        "anomaly_hard_scenes": sorted(set(anomaly_hard)),
        "scenes": entries,
    }
    save_json(stats_path, stats)
    return stats


def build_scene_stats_index(stats_payload: Optional[Dict]) -> Dict[str, Dict]:
    index = {}
    if not isinstance(stats_payload, dict):
        return index
    for row in stats_payload.get("scenes", []):
        scene_path = row.get("scene_path")
        if isinstance(scene_path, str):
            index[str(Path(scene_path).resolve())] = row
    return index


def normalize_summary_schema(summary_path: Path):
    if not summary_path.exists():
        return []
    rows = []
    old_fields = []
    try:
        with summary_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            old_fields = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception:
        return []

    if old_fields == SUMMARY_FIELDS:
        return rows

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in SUMMARY_FIELDS}
            writer.writerow(out)
    return rows


def synthesize_scene_init_for_quarantine(
    scene_path: Path,
    scene_dir: Path,
    stats_path: Path,
    args,
    env_meta: Dict,
    anomaly_info: Dict,
    bootstrap_path: Path,
):
    scene_id = scene_id_from_path(scene_path)
    scene_init_path = scene_dir / "scene_init.json"
    payload = {
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "pipeline_stage": 0,
        "step_name": "stage0_scene_init",
        "status": "QUARANTINED_LARGE_SCENE",
        "run_state": "QUARANTINED",
        "quarantine": True,
        "quarantine_reason": "hard_anomaly_policy",
        "anomaly": anomaly_info,
        "scene_stats_json": str(stats_path),
        "source_root": str(args.source_root),
        "scene_init_ok": False,
        "post_render_ok": False,
        "post_render_fail_reason": "quarantined_no_render",
        "navmesh_status": "FAIL",
        "navmesh_fail_reason": "quarantined_no_navmesh",
        "fallback_used": "no_navmesh_sampling",
        "fallback_plan": "no_navmesh_sampling",
        "navmesh_attempts": [
            {
                "attempt_id": "SKIP_POLICY",
                "settings_summary": None,
                "success": False,
                "error": "quarantined_hard_anomaly",
                "attempt_wall_time_ms": 0.0,
                "navmesh_poly_count": None,
            }
        ],
        "fail_fast_reasons": ["QUARANTINED_HARD_ANOMALY"],
        "init_warnings": [],
        "environment": env_meta,
        "bootstrap_json": str(bootstrap_path),
        "artifacts": {
            "scene_init_json": str(scene_init_path),
            "scene_stats_json": str(stats_path),
        },
    }
    save_json(scene_init_path, payload)


def synthesize_scene_init_for_crash(
    scene_path: Path,
    scene_dir: Path,
    stats_path: Path,
    args,
    env_meta: Dict,
    bootstrap_path: Path,
    run_state: str,
    fail_reason: str,
    worker_exit_code: Optional[int],
    log_path: Path,
):
    scene_id = scene_id_from_path(scene_path)
    scene_init_path = scene_dir / "scene_init.json"
    payload = {
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "pipeline_stage": 0,
        "step_name": "stage0_scene_init",
        "status": run_state,
        "run_state": run_state,
        "quarantine": False,
        "scene_stats_json": str(stats_path),
        "source_root": str(args.source_root),
        "scene_init_ok": False,
        "post_render_ok": False,
        "post_render_fail_reason": "worker_crashed_before_completion",
        "navmesh_status": "FAIL",
        "navmesh_fail_reason": "worker_crashed",
        "fallback_used": "no_navmesh_sampling",
        "fallback_plan": "no_navmesh_sampling",
        "navmesh_attempts": [],
        "fail_fast_reasons": [fail_reason],
        "init_warnings": [],
        "worker_exit_code": worker_exit_code,
        "bootstrap_json": str(bootstrap_path),
        "log_path": str(log_path),
        "environment": env_meta,
        "artifacts": {
            "scene_init_json": str(scene_init_path),
            "scene_stats_json": str(stats_path),
        },
    }
    save_json(scene_init_path, payload)


def parse_scene_init_summary(scene_init_path: Path) -> Dict:
    payload = load_json(scene_init_path)
    if not isinstance(payload, dict):
        return {
            "scene_init_ok": False,
            "run_state": "MISSING",
            "status": "MISSING",
            "quarantine": False,
            "navmesh_status": "MISSING",
            "fallback_plan": "no_navmesh_sampling",
            "fail_reason": "scene_init_json_missing_or_invalid",
        }

    fail_reasons = payload.get("fail_fast_reasons") or []
    fail_reason = fail_reasons[0] if fail_reasons else payload.get("navmesh_fail_reason")
    return {
        "scene_init_ok": bool(payload.get("scene_init_ok", False)),
        "run_state": str(payload.get("run_state") or "UNKNOWN"),
        "status": str(payload.get("status") or "UNKNOWN"),
        "quarantine": bool(payload.get("quarantine", False)),
        "navmesh_status": str(payload.get("navmesh_status") or "UNKNOWN"),
        "fallback_plan": str(payload.get("fallback_plan") or "no_navmesh_sampling"),
        "fail_reason": fail_reason,
    }


def run_stage0_scene_worker(
    scene_path: Path,
    step0_root: Path,
    navmesh_cache_dir: Path,
    stats_path: Path,
    env_meta: Dict,
    args,
    anomaly_info: Dict,
) -> Dict:
    scene_id = scene_id_from_path(scene_path)
    scene_dir = step0_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_init_path = scene_dir / "scene_init.json"
    bootstrap_path = scene_dir / "scene_bootstrap.json"

    write_bootstrap(
        bootstrap_path,
        scene_id,
        scene_path,
        stage="BOOT",
        detail="worker_start",
        extra={"pid": os.getpid(), "anomaly": anomaly_info},
    )

    if anomaly_info.get("hard") and args.quarantine_hard_anomaly:
        write_bootstrap(
            bootstrap_path,
            scene_id,
            scene_path,
            stage="QUARANTINED_HARD_ANOMALY",
            detail="skip_heavy_pipeline",
        )
        synthesize_scene_init_for_quarantine(
            scene_path=scene_path,
            scene_dir=scene_dir,
            stats_path=stats_path,
            args=args,
            env_meta=env_meta,
            anomaly_info=anomaly_info,
            bootstrap_path=bootstrap_path,
        )
        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="DONE_QUARANTINED")
        return {
            "status": "QUARANTINED",
            "scene_init_ok": False,
            "run_state": "QUARANTINED",
            "navmesh_status": "FAIL",
            "fallback_plan": "no_navmesh_sampling",
            "fail_reason": "QUARANTINED_HARD_ANOMALY",
            "scene_init_json": str(scene_init_path),
            "bootstrap_json": str(bootstrap_path),
        }

    fail_fast_reasons = []
    warnings = []
    pre_sim = None
    post_sim = None
    stage_path = None
    scene_instance_path = None
    transformed_scene_path = None
    aabb_pre = None
    aabb_post = None
    aabb_post_expected = None
    transform_plan = None
    pose_pre = None
    pose_post = None
    readback_pre = None
    readback_post = None
    depth_meta_pre = None
    depth_meta_post = None
    navmesh = make_navmesh_skip_result("navmesh_not_run")
    transform_verification = None
    post_scene_source_path = None

    allow_navmesh = True
    navmesh_skip_reason = None
    if anomaly_info.get("soft") and args.skip_navmesh_soft_anomaly:
        allow_navmesh = False
        navmesh_skip_reason = "navmesh_skipped_soft_anomaly"

    try:
        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="LOAD_SIM_PRE")
        pre_sim = build_sim(
            scene_path=scene_path,
            width=args.width,
            height=args.height,
            hfov=args.hfov,
            zfar=args.zfar,
            seed=args.seed,
            with_sensors=True,
        )

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="COMPUTE_AABB_PRE")
        aabb_pre = compute_scene_aabb(pre_sim)
        if not np.isfinite(aabb_pre["max_dim"]) or aabb_pre["max_dim"] <= 1e-6:
            fail_fast_reasons.append("AABB_INVALID_SIZE")

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="COMPUTE_TRANSFORM_PLAN")
        transform_plan = compute_transform_plan(
            aabb=aabb_pre,
            target_min_dim=args.target_min_dim,
            target_max_dim=args.target_max_dim,
            align_floor_to_zero=(not args.disable_floor_align),
        )
        aabb_post_expected = apply_transform_to_aabb(
            aabb_pre,
            scale=float(transform_plan["scale"]),
            translation=list(transform_plan["translation"]),
        )

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="RENDER_PRE_DEBUG")
        pose_pre = build_debug_cam_pose(
            aabb=aabb_pre,
            fov_deg=args.hfov,
            near=args.znear,
            far=args.zfar,
        )
        depth_meta_pre, readback_pre = render_debug_observations(
            sim=pre_sim,
            pose=pose_pre,
            scene_dir=scene_dir,
            tag="pre",
            depth_clip_max_m=args.depth_clip_max_m,
            depth_auto_percentile=args.depth_auto_percentile,
        )
        if depth_meta_pre["valid_ratio"] <= 0.0:
            warnings.append("PRE_RENDER_DEPTH_EMPTY")

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="WRITE_TRANSFORMED_SCENE_GLB")
        transformed_scene_path = write_transformed_scene_glb(
            scene_path=scene_path,
            scene_dir=scene_dir,
            transform_plan=transform_plan,
        )
        post_scene_source_path = transformed_scene_path

        if pre_sim is not None:
            try:
                pre_sim.close()
            except Exception:
                pass
            pre_sim = None

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="LOAD_SIM_POST")
        post_sim = build_sim(
            scene_path=transformed_scene_path,
            width=args.width,
            height=args.height,
            hfov=args.hfov,
            zfar=args.zfar,
            seed=args.seed,
            with_sensors=True,
        )

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="COMPUTE_AABB_POST")
        aabb_post = compute_scene_aabb(post_sim)
        transform_verification = aabb_diff_summary(expected=aabb_post_expected, actual=aabb_post)
        if not transform_verification["matches"]:
            fail_fast_reasons.append("TRANSFORM_VERIFICATION_FAILED")
            transform_plan["applied"] = False
            transform_plan["apply_mode"] = "glb_root_transform_verification_failed"
            write_bootstrap(
                bootstrap_path,
                scene_id,
                scene_path,
                stage="TRANSFORM_VERIFY_FAIL",
                detail="transformed_glb_not_matching_expected_aabb",
                extra=transform_verification,
            )
        else:
            transform_plan["applied"] = True
            transform_plan["apply_mode"] = "glb_root_transform"
            post_scene_source_path = transformed_scene_path

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="RENDER_POST_DEBUG")
        pose_post = build_debug_cam_pose(
            aabb=aabb_post,
            fov_deg=args.hfov,
            near=args.znear,
            far=args.zfar,
        )
        depth_meta_post, readback_post = render_debug_observations(
            sim=post_sim,
            pose=pose_post,
            scene_dir=scene_dir,
            tag="post",
            depth_clip_max_m=args.depth_clip_max_m,
            depth_auto_percentile=args.depth_auto_percentile,
        )
        if depth_meta_post["valid_ratio"] <= 0.0:
            fail_fast_reasons.append("POST_RENDER_DEPTH_EMPTY")

        write_bootstrap(bootstrap_path, scene_id, scene_path, stage="NAVMESH")
        navmesh = run_navmesh_strategy(
            sim=post_sim,
            scene_id=scene_id,
            cache_dir=navmesh_cache_dir,
            max_dim=float((aabb_post or aabb_pre)["max_dim"]),
            allow_navmesh=allow_navmesh,
            skip_reason=navmesh_skip_reason,
        )
    except Exception as exc:
        fail_fast_reasons.append(f"SCENE_INIT_EXCEPTION:{exc}")
        write_bootstrap(
            bootstrap_path,
            scene_id,
            scene_path,
            stage="WORKER_EXCEPTION",
            detail=str(exc),
        )
    finally:
        if pre_sim is not None:
            try:
                pre_sim.close()
            except Exception:
                pass
        if post_sim is not None:
            try:
                post_sim.close()
            except Exception:
                pass

    copy_post_artifacts_to_default(scene_dir)

    scene_init_ok = (
        aabb_pre is not None
        and aabb_pre.get("max_dim", 0.0) > 1e-6
        and depth_meta_post is not None
        and depth_meta_post.get("valid_ratio", 0.0) > 0.0
        and "TRANSFORM_VERIFICATION_FAILED" not in fail_fast_reasons
    )
    post_render_ok = depth_meta_post is not None and depth_meta_post.get("valid_ratio", 0.0) > 0.0

    if not scene_init_ok and "SCENE_INIT_NOT_OK" not in fail_fast_reasons:
        fail_fast_reasons.append("SCENE_INIT_NOT_OK")

    run_state = "OK" if scene_init_ok else "FAIL"
    status = "OK" if scene_init_ok else "FAIL"

    payload = {
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "pipeline_stage": 0,
        "step_name": "stage0_scene_init",
        "status": status,
        "run_state": run_state,
        "quarantine": False,
        "anomaly": anomaly_info,
        "seed": int(args.seed),
        "sensor_config": {
            "width": int(args.width),
            "height": int(args.height),
            "hfov": float(args.hfov),
            "zfar": float(args.zfar),
            "color_sensor": True,
            "depth_sensor": True,
        },
        "source_root": str(args.source_root),
        "scene_stats_json": str(stats_path),
        "aabb_pre": aabb_pre,
        "aabb_post_expected": aabb_post_expected,
        "aabb_post": aabb_post,
        "transform_plan": transform_plan,
        "transform_verification": transform_verification,
        "transform_apply_mode": None if not isinstance(transform_plan, dict) else transform_plan.get("apply_mode"),
        "post_scene_source": str(post_scene_source_path) if post_scene_source_path else None,
        "stage_post_transform": str(stage_path) if stage_path else None,
        "scene_instance_transform": str(scene_instance_path) if scene_instance_path else None,
        "transformed_scene_glb": str(transformed_scene_path) if transformed_scene_path else None,
        "debug_cam_pose_pre": pose_pre,
        "debug_cam_pose_post": pose_post,
        "debug_cam_pose": pose_post if pose_post is not None else pose_pre,
        "debug_cam_pose_readback_pre": readback_pre,
        "debug_cam_pose_readback_post": readback_post,
        "depth_debug_meta_pre": depth_meta_pre,
        "depth_debug_meta_post": depth_meta_post,
        "depth_debug_meta": depth_meta_post if depth_meta_post is not None else depth_meta_pre,
        "post_render_ok": bool(post_render_ok),
        "post_render_fail_reason": None if post_render_ok else "post_render_depth_empty_or_failed",
        "navmesh_status": navmesh["navmesh_status"],
        "navmesh_fail_reason": navmesh["navmesh_fail_reason"],
        "fallback_used": navmesh["fallback_used"],
        "fallback_plan": navmesh["fallback_plan"],
        "navmesh_attempts": navmesh["navmesh_attempts"],
        "navmesh_cache_key": navmesh["navmesh_cache_key"],
        "navmesh_cache_path": navmesh["navmesh_cache_path"],
        "scene_init_ok": bool(scene_init_ok),
        "fail_fast_reasons": fail_fast_reasons,
        "init_warnings": warnings,
        "environment": env_meta,
        "bootstrap_json": str(bootstrap_path),
        "artifacts": {
            "scene_init_json": str(scene_init_path),
            "scene_stats_json": str(stats_path),
            "scene_post_transform": str(transformed_scene_path) if transformed_scene_path else None,
            "stage_post_transform": str(stage_path) if stage_path else None,
            "canonical_scene_source": str(post_scene_source_path) if post_scene_source_path else None,
            "debug_rgb": str(scene_dir / "debug_rgb.png"),
            "debug_depth": str(scene_dir / "debug_depth.png"),
            "debug_depth_clip": str(scene_dir / "debug_depth_clip.png"),
            "debug_depth_npy": str(scene_dir / "debug_depth.npy"),
            "debug_depth_u16": str(scene_dir / "debug_depth_u16.png"),
            "debug_depth_u16_clip": str(scene_dir / "debug_depth_u16_clip.png"),
            "debug_depth_u16_p99": str(scene_dir / "debug_depth_u16_p99.png"),
        },
    }

    write_bootstrap(bootstrap_path, scene_id, scene_path, stage="WRITE_SCENE_INIT")
    save_json(scene_init_path, payload)
    write_bootstrap(bootstrap_path, scene_id, scene_path, stage="DONE")

    fail_reason = fail_fast_reasons[0] if fail_fast_reasons else None
    return {
        "status": status,
        "scene_init_ok": bool(scene_init_ok),
        "run_state": run_state,
        "navmesh_status": navmesh["navmesh_status"],
        "fallback_plan": navmesh["fallback_plan"],
        "fail_reason": fail_reason,
        "scene_init_json": str(scene_init_path),
        "bootstrap_json": str(bootstrap_path),
    }


def discover_scene_inventory(source_root: Path, single_scene: Optional[Path]) -> Tuple[List[Path], Dict[str, int], int]:
    all_scenes = sorted(source_root.rglob("*.glb")) if source_root.exists() else []
    index_map = {str(p.resolve()): idx for idx, p in enumerate(all_scenes, start=1)}

    if single_scene is None:
        return all_scenes, index_map, len(all_scenes)

    scene_resolved = single_scene.resolve()
    if str(scene_resolved) in index_map:
        return [scene_resolved], index_map, len(all_scenes)

    return [scene_resolved], {str(scene_resolved): 1}, 1


def parse_anomaly_for_scene(scene_path: Path, scene_stats_index: Dict[str, Dict], args) -> Dict:
    row = scene_stats_index.get(str(scene_path.resolve())) or {}
    max_dim = row.get("max_dim")
    volume = row.get("aabb_volume")
    max_dim_val = float(max_dim) if isinstance(max_dim, (float, int)) else None
    vol_val = float(volume) if isinstance(volume, (float, int)) else None
    anomaly = classify_anomaly(max_dim=max_dim_val, volume=vol_val, args=args)
    anomaly["from_scene_stats"] = bool(row)
    return anomaly


def build_worker_cmd(script_path: Path, scene_path: Path, args) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-scene",
        str(scene_path),
        "--source-root",
        str(args.source_root),
        "--output-root",
        str(args.output_root),
        "--seed",
        str(args.seed),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--hfov",
        str(args.hfov),
        "--znear",
        str(args.znear),
        "--zfar",
        str(args.zfar),
        "--depth-clip-max-m",
        str(args.depth_clip_max_m),
        "--depth-auto-percentile",
        str(args.depth_auto_percentile),
        "--target-min-dim",
        str(args.target_min_dim),
        "--target-max-dim",
        str(args.target_max_dim),
        "--anomaly-max-dim-soft",
        str(args.anomaly_max_dim_soft),
        "--anomaly-max-dim-hard",
        str(args.anomaly_max_dim_hard),
        "--anomaly-aabb-volume-hard",
        str(args.anomaly_aabb_volume_hard),
        "--skip-stats",
        "--no-resume",
    ]
    if args.disable_floor_align:
        cmd.append("--disable-floor-align")
    if args.quarantine_hard_anomaly:
        cmd.append("--quarantine-hard-anomaly")
    else:
        cmd.append("--no-quarantine-hard-anomaly")
    if args.skip_navmesh_soft_anomaly:
        cmd.append("--skip-navmesh-soft-anomaly")
    else:
        cmd.append("--no-skip-navmesh-soft-anomaly")
    return cmd


def run_scene_with_subprocess(
    scene_path: Path,
    scene_idx: int,
    scene_total: int,
    step0_root: Path,
    navmesh_cache_dir: Path,
    stats_path: Path,
    env_meta: Dict,
    args,
) -> Dict:
    del navmesh_cache_dir
    scene_id = scene_id_from_path(scene_path)
    scene_dir = step0_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_path = scene_dir / "scene_bootstrap.json"
    scene_init_path = scene_dir / "scene_init.json"
    log_dir = step0_root / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scene_id}.log"

    write_bootstrap(
        bootstrap_path,
        scene_id,
        scene_path,
        stage="PARENT_DISPATCH",
        detail="spawn_worker",
        extra={"idx": scene_idx, "total": scene_total},
    )

    worker_cmd = build_worker_cmd(script_path=Path(__file__).resolve(), scene_path=scene_path, args=args)
    t0 = time.time()
    run_state = "UNKNOWN"
    worker_exit_code = None
    timed_out = False
    stdout_text = ""
    stderr_text = ""

    try:
        proc = subprocess.run(
            worker_cmd,
            capture_output=True,
            text=True,
            timeout=float(args.worker_timeout_sec),
            check=False,
        )
        worker_exit_code = int(proc.returncode)
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        run_state = "TIMEOUT"
        worker_exit_code = -9
        stdout_text = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        stderr_text = (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
        stderr_text += f"\nTIMEOUT>{args.worker_timeout_sec}s\n"

    elapsed = time.time() - t0
    with log_path.open("w", encoding="utf-8") as f:
        if stdout_text:
            f.write(stdout_text)
        if stderr_text:
            if stdout_text and not stdout_text.endswith("\n"):
                f.write("\n")
            f.write(stderr_text)

    if timed_out:
        synthesize_scene_init_for_crash(
            scene_path=scene_path,
            scene_dir=scene_dir,
            stats_path=stats_path,
            args=args,
            env_meta=env_meta,
            bootstrap_path=bootstrap_path,
            run_state="TIMEOUT",
            fail_reason="WORKER_TIMEOUT",
            worker_exit_code=worker_exit_code,
            log_path=log_path,
        )
    elif worker_exit_code is not None and worker_exit_code != 0 and not scene_init_path.exists():
        if worker_exit_code < 0:
            run_state = "CRASH_NATIVE"
            fail_reason = f"WORKER_NATIVE_EXIT_{worker_exit_code}"
        else:
            run_state = "WORKER_ERROR"
            fail_reason = f"WORKER_EXIT_NONZERO_{worker_exit_code}"
        synthesize_scene_init_for_crash(
            scene_path=scene_path,
            scene_dir=scene_dir,
            stats_path=stats_path,
            args=args,
            env_meta=env_meta,
            bootstrap_path=bootstrap_path,
            run_state=run_state,
            fail_reason=fail_reason,
            worker_exit_code=worker_exit_code,
            log_path=log_path,
        )

    parsed = parse_scene_init_summary(scene_init_path)

    status = "FAIL"
    if parsed["quarantine"] or parsed["status"].startswith("QUARANTINED"):
        status = "QUARANTINED"
    elif parsed["scene_init_ok"]:
        status = "OK"
    elif parsed["run_state"] in {"CRASH_NATIVE", "TIMEOUT", "WORKER_ERROR"}:
        status = parsed["run_state"]

    return {
        "idx": scene_idx,
        "total": scene_total,
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "status": status,
        "run_state": parsed["run_state"],
        "quarantine": parsed["quarantine"],
        "elapsed_sec": f"{elapsed:.3f}",
        "scene_init_ok": parsed["scene_init_ok"],
        "navmesh_status": parsed["navmesh_status"],
        "fallback_plan": parsed["fallback_plan"],
        "worker_exit_code": worker_exit_code,
        "fail_reason": parsed["fail_reason"],
        "scene_init_json": str(scene_init_path),
        "bootstrap_json": str(bootstrap_path),
        "log_path": str(log_path),
    }


def run_scene_inline(
    scene_path: Path,
    scene_idx: int,
    scene_total: int,
    step0_root: Path,
    navmesh_cache_dir: Path,
    stats_path: Path,
    env_meta: Dict,
    scene_stats_index: Dict[str, Dict],
    args,
) -> Dict:
    anomaly_info = parse_anomaly_for_scene(scene_path=scene_path, scene_stats_index=scene_stats_index, args=args)
    result = run_stage0_scene_worker(
        scene_path=scene_path,
        step0_root=step0_root,
        navmesh_cache_dir=navmesh_cache_dir,
        stats_path=stats_path,
        env_meta=env_meta,
        args=args,
        anomaly_info=anomaly_info,
    )
    return {
        "idx": scene_idx,
        "total": scene_total,
        "scene_id": scene_id_from_path(scene_path),
        "scene_path": str(scene_path),
        "status": result["status"],
        "run_state": result["run_state"],
        "quarantine": result["status"] == "QUARANTINED",
        "elapsed_sec": None,
        "scene_init_ok": result["scene_init_ok"],
        "navmesh_status": result["navmesh_status"],
        "fallback_plan": result["fallback_plan"],
        "worker_exit_code": 0,
        "fail_reason": result["fail_reason"],
        "scene_init_json": result["scene_init_json"],
        "bootstrap_json": result["bootstrap_json"],
        "log_path": "",
    }


def run_worker_entry(args):
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    step0_root = output_root / STEP0_DIRNAME
    step0_root.mkdir(parents=True, exist_ok=True)
    navmesh_cache_dir = step0_root / NAVMESH_CACHE_DIRNAME
    stats_path = step0_root / "scene_stats.json"

    scene_path = args.worker_scene.resolve()
    scene_stats_index = build_scene_stats_index(load_json(stats_path))
    anomaly_info = parse_anomaly_for_scene(scene_path=scene_path, scene_stats_index=scene_stats_index, args=args)
    env_meta = collect_env_meta()

    result = run_stage0_scene_worker(
        scene_path=scene_path,
        step0_root=step0_root,
        navmesh_cache_dir=navmesh_cache_dir,
        stats_path=stats_path,
        env_meta=env_meta,
        args=args,
        anomaly_info=anomaly_info,
    )
    print(
        f"[WORKER] {scene_path.name} | status={result['status']} | run_state={result['run_state']} "
        f"| navmesh={result['navmesh_status']} | fallback={result['fallback_plan']} | fail={result['fail_reason']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene", type=Path, default=None, help="run stage0 for one scene path only")
    parser.add_argument("--max-new", type=int, default=0, help="max number of new scenes to process (0: no limit)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-stats", action="store_true", help="reuse existing scene_stats.json if present")
    parser.add_argument("--stats-only", action="store_true")

    parser.add_argument("--subprocess-isolation", dest="subprocess_isolation", action="store_true", default=True)
    parser.add_argument("--no-subprocess-isolation", dest="subprocess_isolation", action="store_false")
    parser.add_argument("--worker-timeout-sec", type=float, default=240.0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--znear", type=float, default=0.1)
    parser.add_argument("--zfar", type=float, default=1000.0)
    parser.add_argument("--depth-clip-max-m", type=float, default=10.0)
    parser.add_argument("--depth-auto-percentile", type=float, default=99.0)
    parser.add_argument("--target-min-dim", type=float, default=10.0)
    parser.add_argument("--target-max-dim", type=float, default=50.0)
    parser.add_argument("--disable-floor-align", action="store_true")

    parser.add_argument("--anomaly-max-dim-soft", type=float, default=200.0)
    parser.add_argument("--anomaly-max-dim-hard", type=float, default=1000.0)
    parser.add_argument("--anomaly-aabb-volume-hard", type=float, default=1.0e10)
    parser.add_argument("--quarantine-hard-anomaly", dest="quarantine_hard_anomaly", action="store_true", default=True)
    parser.add_argument("--no-quarantine-hard-anomaly", dest="quarantine_hard_anomaly", action="store_false")
    parser.add_argument("--skip-navmesh-soft-anomaly", dest="skip_navmesh_soft_anomaly", action="store_true", default=True)
    parser.add_argument("--no-skip-navmesh-soft-anomaly", dest="skip_navmesh_soft_anomaly", action="store_false")

    parser.add_argument("--worker-scene", type=Path, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker_scene is not None:
        run_worker_entry(args)
        return

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    step0_root = output_root / STEP0_DIRNAME
    step0_root.mkdir(parents=True, exist_ok=True)
    summary_path = step0_root / SUMMARY_PATH_NAME
    navmesh_cache_dir = step0_root / NAVMESH_CACHE_DIRNAME
    stats_path = step0_root / "scene_stats.json"

    single_scene = args.scene.resolve() if args.scene else None
    scene_paths, inventory_index_map, inventory_total = discover_scene_inventory(source_root=source_root, single_scene=single_scene)
    if not scene_paths:
        print(f"No scenes found under: {source_root}")
        return

    if not args.skip_stats or not stats_path.exists():
        print(f"[Stage0] scene stats pass: {len(scene_paths)} scenes")
        run_scene_stats_pass(
            scene_paths=scene_paths,
            source_root=source_root,
            stats_path=stats_path,
            seed=args.seed,
            args=args,
        )
        print(f"[Stage0] wrote scene stats: {stats_path}")
    else:
        print(f"[Stage0] reuse scene stats: {stats_path}")

    if args.stats_only:
        return

    scene_stats_index = build_scene_stats_index(load_json(stats_path))
    existing_rows = normalize_summary_schema(summary_path)
    existing_count = len(existing_rows)
    done_scene_paths = set()
    if args.resume:
        for row in existing_rows:
            sp = (row.get("scene_path") or "").strip()
            if sp:
                done_scene_paths.add(sp)

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    env_meta = collect_env_meta()
    processed_new = 0

    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
            f.flush()

        for loop_i, scene_path in enumerate(scene_paths, start=1):
            scene_path = scene_path.resolve()
            scene_path_str = str(scene_path)
            scene_id = scene_id_from_path(scene_path)
            idx = inventory_index_map.get(scene_path_str, loop_i)
            total = inventory_total

            if scene_path_str in done_scene_paths:
                print(f"[{idx}/{total}] SKIP {scene_id} (already in summary)")
                continue
            if args.max_new > 0 and processed_new >= args.max_new:
                print(f"Reached --max-new={args.max_new}, stop.")
                break

            t0 = time.time()
            if args.subprocess_isolation:
                row = run_scene_with_subprocess(
                    scene_path=scene_path,
                    scene_idx=idx,
                    scene_total=total,
                    step0_root=step0_root,
                    navmesh_cache_dir=navmesh_cache_dir,
                    stats_path=stats_path,
                    env_meta=env_meta,
                    args=args,
                )
            else:
                row = run_scene_inline(
                    scene_path=scene_path,
                    scene_idx=idx,
                    scene_total=total,
                    step0_root=step0_root,
                    navmesh_cache_dir=navmesh_cache_dir,
                    stats_path=stats_path,
                    env_meta=env_meta,
                    scene_stats_index=scene_stats_index,
                    args=args,
                )
                row["elapsed_sec"] = f"{(time.time() - t0):.3f}"

            row_id = existing_count + processed_new + 1
            row_out = {
                "row_id": row_id,
                "idx": row["idx"],
                "total": row["total"],
                "scene_id": row["scene_id"],
                "scene_path": row["scene_path"],
                "status": row["status"],
                "run_state": row["run_state"],
                "quarantine": row["quarantine"],
                "elapsed_sec": row["elapsed_sec"],
                "scene_init_ok": row["scene_init_ok"],
                "navmesh_status": row["navmesh_status"],
                "fallback_plan": row["fallback_plan"],
                "worker_exit_code": row["worker_exit_code"],
                "fail_reason": row["fail_reason"],
                "scene_init_json": row["scene_init_json"],
                "bootstrap_json": row["bootstrap_json"],
                "log_path": row["log_path"],
            }
            writer.writerow(row_out)
            f.flush()

            done_scene_paths.add(scene_path_str)
            processed_new += 1
            print(
                f"[{row['idx']}/{row['total']}] {scene_id} | {row['status']} "
                f"| run_state={row['run_state']} | navmesh={row['navmesh_status']} "
                f"| fallback={row['fallback_plan']} | t={row['elapsed_sec']}s | fail={row['fail_reason']}"
            )

    print(
        f"Stage0 batch done: inventory_total={inventory_total}, newly_processed={processed_new}, "
        f"resume={'on' if args.resume else 'off'}, subprocess_isolation={'on' if args.subprocess_isolation else 'off'}"
    )


if __name__ == "__main__":
    main()
