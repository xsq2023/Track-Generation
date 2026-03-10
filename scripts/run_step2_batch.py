#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from path_defaults import DEFAULT_OUTPUT_ROOT

DEFAULT_STEP1_ROOT = DEFAULT_OUTPUT_ROOT / "step1"
STEP2_DIRNAME = "step2"
SUMMARY_PATH_NAME = "_batch_summary.tsv"
LOG_DIRNAME = "_batch_logs"

SUMMARY_FIELDS = [
    "row_id",
    "idx",
    "total",
    "scene_id",
    "scene_path",
    "step1_report",
    "status",
    "run_state",
    "elapsed_sec",
    "step2_ok",
    "floor_status",
    "floor_method",
    "floor_confidence",
    "fail_reason",
    "step2_report_json",
    "log_path",
    "worker_exit_code",
]


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


def resolve_path_from_base(raw_path: str, base_dir: Path) -> Path:
    p = Path(raw_path)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def validate_scene_source_assets(scene_source: Path) -> Tuple[bool, str, Dict]:
    if not scene_source.exists():
        return False, "scene_source_not_found", {"scene_source": str(scene_source)}

    if scene_source.suffix.lower() == ".glb":
        return True, "ok", {"scene_source": str(scene_source), "type": "glb"}

    if not scene_source.name.endswith(".scene_instance.json"):
        return True, "ok", {"scene_source": str(scene_source), "type": "unknown"}

    scene_payload = load_json(scene_source)
    if not isinstance(scene_payload, dict):
        return False, "scene_instance_invalid_json", {"scene_source": str(scene_source)}

    stage_instance = scene_payload.get("stage_instance") or {}
    template_name = stage_instance.get("template_name")
    if not isinstance(template_name, str) or not template_name.strip():
        return False, "scene_instance_stage_template_missing", {"scene_source": str(scene_source)}

    stage_path = resolve_path_from_base(template_name, scene_source.parent)
    if not stage_path.exists():
        return False, "stage_config_not_found", {"scene_source": str(scene_source), "stage_config": str(stage_path)}

    stage_payload = load_json(stage_path)
    if not isinstance(stage_payload, dict):
        return False, "stage_config_invalid_json", {"scene_source": str(scene_source), "stage_config": str(stage_path)}

    checked_assets = {}
    missing_assets = {}
    for k in ("render_asset", "collision_asset"):
        raw = stage_payload.get(k)
        if not isinstance(raw, str) or not raw.strip():
            continue
        # Primitive assets may not be filesystem paths.
        if raw.startswith(("capsule", "cube", "icosphere", "uvsphere")):
            checked_assets[k] = raw
            continue
        asset_path = resolve_path_from_base(raw, stage_path.parent)
        checked_assets[k] = str(asset_path)
        if not asset_path.exists():
            missing_assets[k] = str(asset_path)

    if missing_assets:
        return (
            False,
            "stage_assets_missing",
            {
                "scene_source": str(scene_source),
                "stage_config": str(stage_path),
                "checked_assets": checked_assets,
                "missing_assets": missing_assets,
            },
        )

    return True, "ok", {"scene_source": str(scene_source), "stage_config": str(stage_path), "checked_assets": checked_assets}


def scene_id_from_path(scene_path: Path) -> str:
    raw = scene_path.stem
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw).strip("_")
    return cleaned or "unknown_scene"


def scene_id_from_step1_report(report_path: Path, payload: Optional[Dict]) -> str:
    if isinstance(payload, dict):
        sid = payload.get("scene_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
        sp = payload.get("scene_path")
        if isinstance(sp, str) and sp.strip():
            return scene_id_from_path(Path(sp))
    return report_path.parent.name


def collect_env_meta() -> Dict:
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
        "module_versions": versions,
    }


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


def compute_scene_aabb(sim: habitat_sim.Simulator) -> Dict:
    root = sim.get_active_scene_graph().get_root_node()
    return aabb_to_dict(root.cumulative_bb)


def resolve_sensor_config(step1_payload: Dict, args) -> Dict:
    base = step1_payload.get("sensor_config") if isinstance(step1_payload.get("sensor_config"), dict) else {}
    width = int(args.width if args.width > 0 else int(base.get("width", 640)))
    height = int(args.height if args.height > 0 else int(base.get("height", 480)))
    hfov = float(args.hfov if args.hfov > 0 else float(base.get("hfov", 90.0)))
    znear = float(args.znear if args.znear > 0 else float(base.get("znear", 0.1)))
    zfar = float(args.zfar if args.zfar > 0 else float(base.get("zfar", 1000.0)))
    return {
        "width": width,
        "height": height,
        "hfov": hfov,
        "znear": znear,
        "zfar": zfar,
    }


def build_sim(scene_source: Path, sensor_cfg: Dict, seed: int, enable_physics: bool) -> habitat_sim.Simulator:
    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": str(scene_source),
            "width": int(sensor_cfg["width"]),
            "height": int(sensor_cfg["height"]),
            "hfov": float(sensor_cfg["hfov"]),
            "zfar": float(sensor_cfg["zfar"]),
            "sensor_height": 0.0,
            "seed": int(seed),
            "silent": True,
            "enable_physics": bool(enable_physics),
            "default_agent_navmesh": False,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": False,
            "frustum_culling": True,
        }
    )
    cfg = make_cfg(settings)
    sim = habitat_sim.Simulator(cfg)
    sim.seed(seed)
    return sim


def yaw_pitch_to_quat_xyzw(yaw_rad: float, pitch_rad: float) -> np.ndarray:
    q_yaw = mn.Quaternion.rotation(mn.Rad(float(yaw_rad)), mn.Vector3(0.0, 1.0, 0.0))
    q_pitch = mn.Quaternion.rotation(mn.Rad(float(pitch_rad)), mn.Vector3(1.0, 0.0, 0.0))
    q = q_yaw * q_pitch
    return np.array([float(q.vector.x), float(q.vector.y), float(q.vector.z), float(q.scalar)], dtype=np.float32)


def set_agent_pose(agent, position: np.ndarray, yaw_rad: float, pitch_rad: float) -> Dict:
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = yaw_pitch_to_quat_xyzw(float(yaw_rad), float(pitch_rad))
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
        "yaw_rad": float(yaw_rad),
        "pitch_rad": float(pitch_rad),
    }


def observe_pose(sim: habitat_sim.Simulator, agent, position: np.ndarray, yaw_rad: float, pitch_rad: float) -> Dict:
    readback = set_agent_pose(agent=agent, position=position, yaw_rad=yaw_rad, pitch_rad=pitch_rad)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["color_sensor"])
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = np.asarray(rgb, dtype=np.uint8)
    depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
    return {
        "rgb": rgb,
        "depth": depth,
        "pose_readback": readback,
    }


def normalize_depth_vis(depth: np.ndarray, depth_clip_max_m: float) -> np.ndarray:
    finite = np.isfinite(depth)
    valid = finite & (depth > 0.0)
    if np.any(valid):
        vals = depth[valid]
        dmin = float(np.percentile(vals, 1.0))
        dmax = float(np.percentile(vals, 99.0))
        if dmax <= dmin:
            dmax = dmin + 1e-3
        vis = np.clip((depth - dmin) / (dmax - dmin), 0.0, 1.0)
        return np.round(vis * 255.0).astype(np.uint8)
    clip = np.clip(np.where(finite, depth, 0.0), 0.0, depth_clip_max_m)
    return np.round(clip / max(depth_clip_max_m, 1e-6) * 255.0).astype(np.uint8)


def save_step2_render_artifacts(scene_dir: Path, rgb_before: np.ndarray, depth_before: np.ndarray, rgb_after: np.ndarray):
    scene_dir.mkdir(parents=True, exist_ok=True)
    before_rgb_path = scene_dir / "step2_floor_rgb_before.png"
    before_depth_path = scene_dir / "step2_floor_depth_before.png"
    before_depth_npy = scene_dir / "step2_floor_depth_before.npy"
    after_rgb_path = scene_dir / "step2_floor_rgb.png"

    Image.fromarray(rgb_before).save(before_rgb_path)
    Image.fromarray(normalize_depth_vis(depth_before, depth_clip_max_m=10.0)).save(before_depth_path)
    np.save(before_depth_npy, depth_before)
    Image.fromarray(rgb_after).save(after_rgb_path)
    return {
        "step2_floor_rgb_before": str(before_rgb_path),
        "step2_floor_depth_before": str(before_depth_path),
        "step2_floor_depth_before_npy": str(before_depth_npy),
        "step2_floor_rgb": str(after_rgb_path),
    }


def quat_wxyz_to_rotmat(q_wxyz: List[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w /= n
    x /= n
    y /= n
    z /= n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_intrinsics(width: int, height: int, hfov_deg: float) -> Dict:
    fx = (float(width) * 0.5) / math.tan(math.radians(float(hfov_deg)) * 0.5)
    fy = fx
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": int(width),
        "height": int(height),
        "hfov_deg": float(hfov_deg),
    }


def backproject_depth_roi(
    depth: np.ndarray,
    K: Dict,
    T_wc: np.ndarray,
    roi_y0_ratio: float,
    roi_y1_ratio: float,
    depth_min_m: float,
    enforce_depth_valid_max: bool,
    depth_max_m: float,
) -> Dict:
    h, w = depth.shape
    y0 = int(np.clip(round(h * roi_y0_ratio), 0, h - 1))
    y1 = int(np.clip(round(h * roi_y1_ratio), y0 + 1, h))

    roi = depth[y0:y1, :]
    valid = np.isfinite(roi) & (roi > float(depth_min_m))
    if enforce_depth_valid_max:
        valid &= roi < float(depth_max_m)

    if not np.any(valid):
        return {
            "roi_shape": [int(y1 - y0), int(w)],
            "roi_y_range": [int(y0), int(y1)],
            "valid_count": 0,
            "cam_points": np.zeros((0, 3), dtype=np.float64),
            "world_points": np.zeros((0, 3), dtype=np.float64),
            "uv": np.zeros((0, 2), dtype=np.int32),
        }

    vv, uu = np.where(valid)
    vv = vv + y0
    d = depth[vv, uu].astype(np.float64)

    x = (uu.astype(np.float64) - float(K["cx"])) / float(K["fx"]) * d
    y = -(vv.astype(np.float64) - float(K["cy"])) / float(K["fy"]) * d
    z = -d
    cam_pts = np.stack([x, y, z], axis=1)

    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    world_pts = (R_wc @ cam_pts.T).T + t_wc[None, :]

    uv = np.stack([uu.astype(np.int32), vv.astype(np.int32)], axis=1)
    return {
        "roi_shape": [int(y1 - y0), int(w)],
        "roi_y_range": [int(y0), int(y1)],
        "valid_count": int(cam_pts.shape[0]),
        "cam_points": cam_pts,
        "world_points": world_pts,
        "uv": uv,
    }


def transform_self_check(cam_pts: np.ndarray, T_wc: np.ndarray, uv: np.ndarray, K: Dict, p_cam: np.ndarray) -> Dict:
    if cam_pts.shape[0] < 20:
        return {
            "ok": False,
            "reason": "too_few_points_for_transform_check",
            "closure_median_m": None,
            "closure_p95_m": None,
            "center_ray_cos": None,
        }

    n = min(200, cam_pts.shape[0])
    sample_idx = np.linspace(0, cam_pts.shape[0] - 1, n).astype(np.int32)
    c = cam_pts[sample_idx]
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    w = (R @ c.T).T + t[None, :]
    c_round = (R.T @ (w - t[None, :]).T).T
    err = np.linalg.norm(c - c_round, axis=1)
    med = float(np.median(err))
    p95 = float(np.percentile(err, 95.0))

    center_u = int(round(float(K["cx"])))
    center_v = int(round(float(K["cy"])))
    dist2 = (uv[:, 0] - center_u) ** 2 + (uv[:, 1] - center_v) ** 2
    i0 = int(np.argmin(dist2))
    center_world = (R @ cam_pts[i0].reshape(3, 1)).reshape(3) + t
    forward_world = (R @ np.array([0.0, 0.0, -1.0], dtype=np.float64).reshape(3, 1)).reshape(3)
    vec = center_world - p_cam
    nv = np.linalg.norm(vec)
    nf = np.linalg.norm(forward_world)
    center_cos = float(np.dot(vec / max(nv, 1e-9), forward_world / max(nf, 1e-9)))

    ok = med <= 1e-3 and center_cos >= 0.95
    return {
        "ok": bool(ok),
        "reason": None if ok else "transform_inconsistent",
        "closure_median_m": med,
        "closure_p95_m": p95,
        "center_ray_cos": center_cos,
    }


def fit_plane_from_points(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    _, _, vt = np.linalg.svd(points - centroid[None, :], full_matrices=False)
    n = vt[-1]
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return None
    n = n / nn
    d = -float(np.dot(n, centroid))
    return n, d


def ransac_plane(points: np.ndarray, max_iter: int, dist_thresh: float, rng: np.random.Generator) -> Optional[Dict]:
    n_pts = points.shape[0]
    if n_pts < 3:
        return None
    best_inliers = None
    best_count = 0
    best_model = None

    for _ in range(int(max_iter)):
        idx = rng.choice(n_pts, size=3, replace=False)
        p = points[idx]
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        n = np.cross(v1, v2)
        nn = float(np.linalg.norm(n))
        if nn < 1e-10:
            continue
        n = n / nn
        d = -float(np.dot(n, p[0]))
        residual = np.abs(points @ n + d)
        inlier = residual <= float(dist_thresh)
        c = int(np.sum(inlier))
        if c > best_count:
            best_count = c
            best_inliers = inlier
            best_model = (n, d)

    if best_model is None or best_inliers is None or int(np.sum(best_inliers)) < 3:
        return None

    refined = fit_plane_from_points(points[best_inliers])
    if refined is None:
        return None
    n, d = refined
    residual = np.abs(points @ n + d)
    inlier = residual <= float(dist_thresh)
    return {
        "n": n,
        "d": float(d),
        "inlier_mask": inlier,
        "residual": residual,
    }


def inlier_coverage_ratio(
    uv_slice: np.ndarray,
    inlier_mask: np.ndarray,
    roi_y_range: List[int],
    grid_w: int,
    grid_h: int,
    image_w: int,
) -> float:
    if uv_slice.shape[0] == 0:
        return 0.0
    y0, y1 = int(roi_y_range[0]), int(roi_y_range[1])
    roi_h = max(y1 - y0, 1)

    gx_valid = np.clip((uv_slice[:, 0].astype(np.float64) / max(float(image_w), 1.0) * grid_w).astype(np.int32), 0, grid_w - 1)
    gy_valid = np.clip((((uv_slice[:, 1] - y0).astype(np.float64) / max(float(roi_h), 1.0)) * grid_h).astype(np.int32), 0, grid_h - 1)
    valid_cells = set((int(x), int(y)) for x, y in zip(gx_valid, gy_valid))
    if not valid_cells:
        return 0.0

    uv_in = uv_slice[inlier_mask]
    gx_in = np.clip((uv_in[:, 0].astype(np.float64) / max(float(image_w), 1.0) * grid_w).astype(np.int32), 0, grid_w - 1)
    gy_in = np.clip((((uv_in[:, 1] - y0).astype(np.float64) / max(float(roi_h), 1.0)) * grid_h).astype(np.int32), 0, grid_h - 1)
    in_cells = set((int(x), int(y)) for x, y in zip(gx_in, gy_in))

    return float(len(in_cells) / max(len(valid_cells), 1))


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def floor_confidence_ransac(inlier_ratio: float, residual_p95: float, cam_height: float, args) -> float:
    term1 = float(inlier_ratio)
    term2 = 1.0 - float(residual_p95) / max(float(args.max_residual_m), 1e-6)
    term3 = 1.0 - abs(float(cam_height) - float(args.conf_h_mid_m)) / max(float(args.conf_h_span_m), 1e-6)
    return clamp01(float(args.conf_w1) * term1 + float(args.conf_w2) * term2 + float(args.conf_w3) * term3)


def floor_confidence_ray(ray_hit_count: int, ray_expected_hits: int, ray_mad: float) -> float:
    term1 = float(ray_hit_count) / max(float(ray_expected_hits), 1.0)
    term2 = 1.0 - float(ray_mad) / 0.05
    return clamp01(0.6 * term1 + 0.4 * term2)


def estimate_floor_ransac(
    pc: Dict,
    p_cam: np.ndarray,
    rng: np.random.Generator,
    args,
    image_w: int,
) -> Dict:
    world_pts = pc["world_points"]
    uv = pc["uv"]
    total_valid = int(world_pts.shape[0])
    sampled_points = min(total_valid, int(args.ransac_max_points))
    if sampled_points < int(args.ransac_min_points):
        return {
            "ok": False,
            "fail_reason": "insufficient_depth_points",
            "fail_reason_detail": f"sampled_points={sampled_points} < min_points={int(args.ransac_min_points)}",
            "ransac_attempts": [],
        }

    if total_valid > sampled_points:
        idx_all = rng.choice(total_valid, size=sampled_points, replace=False)
        pts = world_pts[idx_all]
        uv_pts = uv[idx_all]
    else:
        pts = world_pts
        uv_pts = uv

    y_vals = pts[:, 1]
    quantiles = [float(x.strip()) for x in str(args.ransac_low_height_quantiles).split(",") if x.strip()]
    if not quantiles:
        quantiles = [0.2, 0.35, 0.5, 1.0]
    if 1.0 not in quantiles:
        quantiles.append(1.0)

    attempts = []
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for q in quantiles:
        y_thr = float(np.percentile(y_vals, q * 100.0))
        mask_q = y_vals <= y_thr
        pts_q = pts[mask_q]
        uv_q = uv_pts[mask_q]
        if pts_q.shape[0] < int(args.ransac_min_points):
            attempts.append(
                {
                    "quantile": q,
                    "points": int(pts_q.shape[0]),
                    "status": "insufficient_points",
                }
            )
            continue

        model = ransac_plane(
            points=pts_q,
            max_iter=int(args.ransac_max_iter),
            dist_thresh=float(args.ransac_dist_thresh_m),
            rng=rng,
        )
        if model is None:
            attempts.append({"quantile": q, "points": int(pts_q.shape[0]), "status": "ransac_no_plane"})
            continue

        n = np.asarray(model["n"], dtype=np.float64)
        d = float(model["d"])
        if n[1] < 0.0:
            n = -n
            d = -d

        n_norm = float(np.linalg.norm(n))
        if abs(n_norm - 1.0) > float(args.normal_unit_tol):
            attempts.append(
                {
                    "quantile": q,
                    "points": int(pts_q.shape[0]),
                    "status": "normal_not_unit",
                    "n_norm": n_norm,
                }
            )
            continue

        dot_up = float(np.dot(n, up))
        ny = float(n[1])
        inlier_mask = np.asarray(model["inlier_mask"], dtype=bool)
        residual = np.asarray(model["residual"], dtype=np.float64)
        inlier_ratio = float(np.mean(inlier_mask))
        residual_p50 = float(np.percentile(residual, 50.0))
        residual_p95 = float(np.percentile(residual, 95.0))
        cam_height = float(np.dot(n, p_cam) + d)
        coverage = inlier_coverage_ratio(
            uv_slice=uv_q,
            inlier_mask=inlier_mask,
            roi_y_range=pc["roi_y_range"],
            grid_w=int(args.coverage_grid_w),
            grid_h=int(args.coverage_grid_h),
            image_w=int(image_w),
        )

        ok = True
        fail_items = []
        if ny < float(args.min_ny):
            ok = False
            fail_items.append(f"n_y={ny:.5f} < min_ny={float(args.min_ny):.5f}")
        if dot_up < float(args.min_normal_dot_up):
            ok = False
            fail_items.append(f"dot_up={dot_up:.5f} < min_dot={float(args.min_normal_dot_up):.5f}")
        if inlier_ratio < float(args.min_inlier_ratio):
            ok = False
            fail_items.append(f"inlier_ratio={inlier_ratio:.5f} < min_inlier_ratio={float(args.min_inlier_ratio):.5f}")
        if residual_p95 > float(args.max_residual_m):
            ok = False
            fail_items.append(f"residual_p95={residual_p95:.5f} > max_residual={float(args.max_residual_m):.5f}")
        if coverage < float(args.min_inlier_coverage):
            ok = False
            fail_items.append(f"inlier_coverage={coverage:.5f} < min_coverage={float(args.min_inlier_coverage):.5f}")
        if cam_height < float(args.cam_height_min_m) or cam_height > float(args.cam_height_max_m):
            ok = False
            fail_items.append(
                f"cam_height_above_floor={cam_height:.5f} not in [{float(args.cam_height_min_m):.3f},{float(args.cam_height_max_m):.3f}]"
            )

        attempt_row = {
            "quantile": q,
            "points": int(pts_q.shape[0]),
            "status": "ok" if ok else "threshold_fail",
            "n": n.tolist(),
            "d": d,
            "dot_up": dot_up,
            "n_y": ny,
            "inlier_ratio": inlier_ratio,
            "plane_residual_p50": residual_p50,
            "plane_residual_p95": residual_p95,
            "cam_height_above_floor": cam_height,
            "inlier_coverage": coverage,
            "fail_items": fail_items,
        }
        attempts.append(attempt_row)

        if ok:
            floor_height = float(-d / max(ny, 1e-9))
            conf = floor_confidence_ransac(inlier_ratio=inlier_ratio, residual_p95=residual_p95, cam_height=cam_height, args=args)
            return {
                "ok": True,
                "floor_status": "OK" if dot_up >= 0.9 else "SLOPE_OK",
                "floor_method": "ransac",
                "floor_plane": [float(n[0]), float(n[1]), float(n[2]), float(d)],
                "floor_normal": [float(n[0]), float(n[1]), float(n[2])],
                "floor_height": floor_height,
                "cam_height_above_floor": cam_height,
                "inlier_ratio": inlier_ratio,
                "plane_residual_p50": residual_p50,
                "plane_residual_p95": residual_p95,
                "inlier_coverage": coverage,
                "ransac_low_height_quantile_used": q,
                "ransac_attempts": attempts,
                "floor_confidence": conf,
                "valid_points_in_roi": total_valid,
                "sampled_points": int(sampled_points),
            }

    return {
        "ok": False,
        "fail_reason": "ransac_no_plane",
        "fail_reason_detail": "all_ransac_quantiles_failed",
        "ransac_attempts": attempts,
        "valid_points_in_roi": total_valid,
        "sampled_points": int(sampled_points),
    }


def first_hit(sim: habitat_sim.Simulator, origin: np.ndarray, direction: np.ndarray) -> Optional[Dict]:
    try:
        ray = habitat_sim.geo.Ray()
        ray.origin = mn.Vector3(float(origin[0]), float(origin[1]), float(origin[2]))
        ray.direction = mn.Vector3(float(direction[0]), float(direction[1]), float(direction[2]))
        result = sim.cast_ray(ray=ray)
        if not result.has_hits():
            return None
        hit = result.hits[0]
        return {
            "point": np.array([float(hit.point[0]), float(hit.point[1]), float(hit.point[2])], dtype=np.float64),
            "normal": np.array([float(hit.normal[0]), float(hit.normal[1]), float(hit.normal[2])], dtype=np.float64),
            "distance": float(hit.ray_distance),
            "object_id": int(hit.object_id),
        }
    except Exception:
        return None


def ray_grid_offsets(grid_name: str, offset_m: float) -> List[Tuple[float, float]]:
    off = float(offset_m)
    if grid_name == "cross5":
        return [(0.0, 0.0), (off, 0.0), (-off, 0.0), (0.0, off), (0.0, -off)]
    return [
        (-off, -off),
        (0.0, -off),
        (off, -off),
        (-off, 0.0),
        (0.0, 0.0),
        (off, 0.0),
        (-off, off),
        (0.0, off),
        (off, off),
    ]


def ray_preflight(sim: habitat_sim.Simulator, p_cam: np.ndarray, args) -> Dict:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    offsets = ray_grid_offsets("cross5", float(args.ray_offset_m))
    hits = []
    for dx, dz in offsets:
        origin = p_cam + up * 0.5 + np.array([dx, 0.0, dz], dtype=np.float64)
        hit = first_hit(sim=sim, origin=origin, direction=-up)
        if hit is not None:
            hits.append(hit)
    hit_ratio = float(len(hits) / max(len(offsets), 1))
    return {
        "grid": "cross5",
        "hit_count": int(len(hits)),
        "expected_hits": int(len(offsets)),
        "hit_ratio": hit_ratio,
        "first_hit_distance": hits[0]["distance"] if hits else None,
        "ok": bool(hit_ratio >= float(args.ray_preflight_min_hit_ratio)),
        "fail_reason": None if hit_ratio >= float(args.ray_preflight_min_hit_ratio) else "ray_preflight_low_hit_ratio",
    }


def ray_fallback_from_cast(
    sim: habitat_sim.Simulator,
    p_cam: np.ndarray,
    args,
) -> Dict:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    offsets = ray_grid_offsets(str(args.ray_grid), float(args.ray_offset_m))
    heights = []
    normals = []
    for dx, dz in offsets:
        origin = p_cam + up * 0.05 + np.array([dx, 0.0, dz], dtype=np.float64)
        hit = first_hit(sim=sim, origin=origin, direction=-up)
        if hit is None:
            continue
        dot_up = float(np.dot(hit["normal"], up))
        if dot_up < float(args.ray_min_normal_dot):
            continue
        heights.append(float(hit["point"][1]))
        normals.append(hit["normal"])

    hit_count = len(heights)
    expected = len(offsets)
    if hit_count <= 0:
        return {
            "ok": False,
            "fail_reason": "ray_no_hit",
            "fail_reason_detail": "ray_hit_count=0",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_method": "ray_fallback",
        }

    h_arr = np.asarray(heights, dtype=np.float64)
    h_med = float(np.median(h_arr))
    h_mad = float(np.median(np.abs(h_arr - h_med)))
    cam_height = float(p_cam[1] - h_med)

    if hit_count < int(args.ray_min_hits):
        return {
            "ok": False,
            "fail_reason": "ray_no_hit",
            "fail_reason_detail": f"ray_hit_count={hit_count} < min_hits={int(args.ray_min_hits)}",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_hit_height_median": h_med,
            "ray_hit_height_mad": h_mad,
            "ray_method": "ray_fallback",
        }

    if h_mad > float(args.ray_max_mad_m):
        return {
            "ok": False,
            "fail_reason": "ray_height_unstable",
            "fail_reason_detail": f"ray_hit_height_mad={h_mad:.5f} > max_mad={float(args.ray_max_mad_m):.5f}",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_hit_height_median": h_med,
            "ray_hit_height_mad": h_mad,
            "ray_method": "ray_fallback",
        }

    if cam_height < float(args.cam_height_min_m) or cam_height > float(args.cam_height_max_m):
        return {
            "ok": False,
            "fail_reason": "camera_height_out_of_range",
            "fail_reason_detail": f"cam_height_above_floor={cam_height:.5f} not in [{float(args.cam_height_min_m):.3f},{float(args.cam_height_max_m):.3f}]",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_hit_height_median": h_med,
            "ray_hit_height_mad": h_mad,
            "ray_method": "ray_fallback",
        }

    conf = floor_confidence_ray(ray_hit_count=hit_count, ray_expected_hits=expected, ray_mad=h_mad)
    return {
        "ok": True,
        "floor_status": "FALLBACK_OK",
        "floor_method": "ray_fallback",
        "floor_plane": [0.0, 1.0, 0.0, -h_med],
        "floor_normal": [0.0, 1.0, 0.0],
        "floor_height": h_med,
        "cam_height_above_floor": cam_height,
        "ray_hit_count": int(hit_count),
        "ray_expected_hits": int(expected),
        "ray_hit_height_median": h_med,
        "ray_hit_height_mad": h_mad,
        "ray_grid": str(args.ray_grid),
        "ray_offset_m": float(args.ray_offset_m),
        "floor_confidence": conf,
        "ray_method": "ray_fallback",
    }


def ray_fallback_depth_proxy(pc: Dict, p_cam: np.ndarray, args) -> Dict:
    world_pts = pc["world_points"]
    if world_pts.shape[0] <= 0:
        return {
            "ok": False,
            "fail_reason": "ray_no_hit",
            "fail_reason_detail": "depth_proxy_empty_world_points",
            "ray_hit_count": 0,
            "ray_expected_hits": len(ray_grid_offsets(str(args.ray_grid), float(args.ray_offset_m))),
            "ray_method": "ray_fallback_depth_proxy",
        }

    offsets = ray_grid_offsets(str(args.ray_grid), float(args.ray_offset_m))
    heights = []
    expected = len(offsets)
    radius = float(args.proxy_radius_m)
    low_q = float(args.proxy_low_quantile)
    for dx, dz in offsets:
        origin_xz = np.array([p_cam[0] + dx, p_cam[2] + dz], dtype=np.float64)
        d2 = np.sum((world_pts[:, [0, 2]] - origin_xz[None, :]) ** 2, axis=1)
        near = world_pts[d2 <= radius * radius]
        if near.shape[0] <= 0:
            continue
        h = float(np.percentile(near[:, 1], low_q * 100.0))
        heights.append(h)

    hit_count = len(heights)
    if hit_count <= 0:
        return {
            "ok": False,
            "fail_reason": "ray_no_hit",
            "fail_reason_detail": "depth_proxy_no_local_points",
            "ray_hit_count": 0,
            "ray_expected_hits": int(expected),
            "ray_method": "ray_fallback_depth_proxy",
        }

    h_arr = np.asarray(heights, dtype=np.float64)
    h_med = float(np.median(h_arr))
    h_mad = float(np.median(np.abs(h_arr - h_med)))
    cam_height = float(p_cam[1] - h_med)

    if hit_count < int(args.ray_min_hits):
        return {
            "ok": False,
            "fail_reason": "ray_no_hit",
            "fail_reason_detail": f"depth_proxy_hit_count={hit_count} < min_hits={int(args.ray_min_hits)}",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_hit_height_median": h_med,
            "ray_hit_height_mad": h_mad,
            "ray_method": "ray_fallback_depth_proxy",
        }

    if h_mad > float(args.ray_max_mad_m):
        return {
            "ok": False,
            "fail_reason": "ray_height_unstable",
            "fail_reason_detail": f"depth_proxy_mad={h_mad:.5f} > max_mad={float(args.ray_max_mad_m):.5f}",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_hit_height_median": h_med,
            "ray_hit_height_mad": h_mad,
            "ray_method": "ray_fallback_depth_proxy",
        }

    if cam_height < float(args.cam_height_min_m) or cam_height > float(args.cam_height_max_m):
        return {
            "ok": False,
            "fail_reason": "camera_height_out_of_range",
            "fail_reason_detail": f"cam_height_above_floor={cam_height:.5f} not in [{float(args.cam_height_min_m):.3f},{float(args.cam_height_max_m):.3f}]",
            "ray_hit_count": int(hit_count),
            "ray_expected_hits": int(expected),
            "ray_hit_height_median": h_med,
            "ray_hit_height_mad": h_mad,
            "ray_method": "ray_fallback_depth_proxy",
        }

    conf = floor_confidence_ray(ray_hit_count=hit_count, ray_expected_hits=expected, ray_mad=h_mad)
    return {
        "ok": True,
        "floor_status": "FALLBACK_OK",
        "floor_method": "ray_fallback_depth_proxy",
        "floor_plane": [0.0, 1.0, 0.0, -h_med],
        "floor_normal": [0.0, 1.0, 0.0],
        "floor_height": h_med,
        "cam_height_above_floor": cam_height,
        "ray_hit_count": int(hit_count),
        "ray_expected_hits": int(expected),
        "ray_hit_height_median": h_med,
        "ray_hit_height_mad": h_mad,
        "ray_grid": str(args.ray_grid),
        "ray_offset_m": float(args.ray_offset_m),
        "floor_confidence": conf,
        "ray_method": "ray_fallback_depth_proxy",
    }


def rgb_l1_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def depth_shift_stat(depth_before: np.ndarray, depth_after: np.ndarray, roi_y0_ratio: float, roi_y1_ratio: float) -> Dict:
    h, _ = depth_before.shape
    y0 = int(np.clip(round(h * roi_y0_ratio), 0, h - 1))
    y1 = int(np.clip(round(h * roi_y1_ratio), y0 + 1, h))
    b = depth_before[y0:y1, :]
    a = depth_after[y0:y1, :]
    valid = np.isfinite(b) & np.isfinite(a) & (b > 0.0) & (a > 0.0)
    if not np.any(valid):
        return {"median": None, "p95": None}
    d = (a - b)[valid]
    return {
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95.0)),
    }


def dependency_gate(step1_payload: Dict) -> Tuple[bool, str]:
    if not bool(step1_payload.get("step1_ok", False)):
        return False, "STEP1_NOT_READY"
    if str(step1_payload.get("status") or "").upper() != "OK":
        return False, "STEP1_STATUS_NOT_OK"
    pose = step1_payload.get("chosen_start_pose")
    if not isinstance(pose, dict) or not isinstance(pose.get("position"), list):
        return False, "STEP1_POSE_MISSING"
    return True, "OK"


def run_step2_scene_worker(step1_report_path: Path, step2_root: Path, args, env_meta: Dict) -> Dict:
    step1_payload = load_json(step1_report_path)
    scene_id = scene_id_from_step1_report(report_path=step1_report_path, payload=step1_payload)
    scene_dir = step2_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / "step2_floor_report.json"

    if not isinstance(step1_payload, dict):
        payload = {
            "scene_id": scene_id,
            "pipeline_stage": 2,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "fail_reason": "STEP1_REPORT_MISSING_OR_INVALID",
            "step1_report": str(step1_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": "",
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "floor_confidence": 0.0,
            "fail_reason": "STEP1_REPORT_MISSING_OR_INVALID",
            "step2_report_json": str(report_path),
        }

    dep_ok, dep_reason = dependency_gate(step1_payload)
    scene_path = str(step1_payload.get("scene_path") or "")
    if not dep_ok:
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 2,
            "status": "SKIP",
            "run_state": "SKIP",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "fail_reason": dep_reason,
            "step1_report": str(step1_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "SKIP",
            "run_state": "SKIP",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "floor_confidence": 0.0,
            "fail_reason": dep_reason,
            "step2_report_json": str(report_path),
        }

    scene_source_raw = step1_payload.get("scene_source")
    if not isinstance(scene_source_raw, str) or not scene_source_raw.strip():
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 2,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "fail_reason": "STEP1_SCENE_SOURCE_MISSING",
            "step1_report": str(step1_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "floor_confidence": 0.0,
            "fail_reason": "STEP1_SCENE_SOURCE_MISSING",
            "step2_report_json": str(report_path),
        }

    scene_source = Path(scene_source_raw)
    if not scene_source.exists():
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 2,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "fail_reason": "STEP1_SCENE_SOURCE_NOT_FOUND",
            "step1_report": str(step1_report_path),
            "scene_source": str(scene_source),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "floor_confidence": 0.0,
            "fail_reason": "STEP1_SCENE_SOURCE_NOT_FOUND",
            "step2_report_json": str(report_path),
        }

    source_ok, source_reason, source_detail = validate_scene_source_assets(scene_source=scene_source)
    if not source_ok:
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 2,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "fail_reason": "STEP2_SCENE_SOURCE_INVALID",
            "fail_reason_detail": source_reason,
            "scene_source_validation": source_detail,
            "step1_report": str(step1_report_path),
            "scene_source": str(scene_source),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "floor_confidence": 0.0,
            "fail_reason": "STEP2_SCENE_SOURCE_INVALID",
            "step2_report_json": str(report_path),
        }

    sensor_cfg = resolve_sensor_config(step1_payload=step1_payload, args=args)
    pose0 = step1_payload.get("chosen_start_pose") or {}
    p0 = np.asarray(pose0.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    yaw0 = float(pose0.get("yaw_rad", 0.0))
    pitch0 = -math.radians(float(args.lookdown_pitch_deg)) if bool(args.floor_init_lookdown) else 0.0

    sim = None
    t0 = time.time()
    timings = {}
    report = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 2,
        "step_name": "step2_floor_estimation",
        "step1_report": str(step1_report_path),
        "scene_source": str(scene_source),
        "sensor_config": sensor_cfg,
        "floor_init_lookdown_applied": bool(args.floor_init_lookdown),
        "floor_init_lookdown_delta_y_m": float(args.lookdown_delta_y_m),
        "params": {
            "ransac_low_height_quantiles": str(args.ransac_low_height_quantiles),
            "ransac_max_points": int(args.ransac_max_points),
            "ransac_min_points": int(args.ransac_min_points),
            "ransac_max_iter": int(args.ransac_max_iter),
            "ransac_dist_thresh_m": float(args.ransac_dist_thresh_m),
            "min_inlier_ratio": float(args.min_inlier_ratio),
            "max_residual_m": float(args.max_residual_m),
            "min_inlier_coverage": float(args.min_inlier_coverage),
            "min_normal_dot_up": float(args.min_normal_dot_up),
            "min_ny": float(args.min_ny),
            "cam_height_range_m": [float(args.cam_height_min_m), float(args.cam_height_max_m)],
            "ray_grid": str(args.ray_grid),
            "ray_offset_m": float(args.ray_offset_m),
            "ray_min_hits": int(args.ray_min_hits),
            "ray_max_mad_m": float(args.ray_max_mad_m),
            "ray_preflight_min_hit_ratio": float(args.ray_preflight_min_hit_ratio),
        },
        "environment": env_meta,
    }

    status = "FAIL"
    run_state = "FAIL"
    step2_ok = False
    floor_status = "FAIL"
    floor_method = "none"
    floor_confidence = 0.0
    fail_reason = "UNKNOWN"
    fail_reason_detail = None
    warnings = []

    floor_plane = None
    floor_normal = None
    floor_height = None
    cam_height = None
    scene_aabb = None
    ray_preflight_info = None
    ray_result = None
    ransac_result = None
    selected_floor_result = None
    ransac_attempts = []
    artifacts = {}

    try:
        ts = time.time()
        scene_seed = int((int(args.seed) + (abs(hash(scene_id)) % 2147483647)) % 2147483647)
        sim = build_sim(scene_source=scene_source, sensor_cfg=sensor_cfg, seed=scene_seed, enable_physics=(not args.disable_physics))
        agent = sim.initialize_agent(0)
        scene_aabb = compute_scene_aabb(sim)
        timings["t_load_scene"] = float(time.time() - ts)

        ts = time.time()
        obs_before = observe_pose(
            sim=sim,
            agent=agent,
            position=p0.astype(np.float32),
            yaw_rad=yaw0,
            pitch_rad=pitch0,
        )
        p_cam_before = np.asarray(obs_before["pose_readback"]["position"], dtype=np.float64)
        q_wxyz = obs_before["pose_readback"]["rotation_quat_wxyz"]
        R_wc = quat_wxyz_to_rotmat(q_wxyz)
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = p_cam_before
        K = camera_intrinsics(
            width=int(sensor_cfg["width"]),
            height=int(sensor_cfg["height"]),
            hfov_deg=float(sensor_cfg["hfov"]),
        )
        timings["t_render_before"] = float(time.time() - ts)

        ts = time.time()
        pc = backproject_depth_roi(
            depth=np.asarray(obs_before["depth"], dtype=np.float32),
            K=K,
            T_wc=T_wc,
            roi_y0_ratio=float(args.roi_y0_ratio),
            roi_y1_ratio=float(args.roi_y1_ratio),
            depth_min_m=float(args.depth_valid_min_m),
            enforce_depth_valid_max=bool(args.enforce_depth_valid_max),
            depth_max_m=float(args.depth_valid_max_m),
        )

        transform_check = transform_self_check(
            cam_pts=pc["cam_points"],
            T_wc=T_wc,
            uv=pc["uv"],
            K=K,
            p_cam=p_cam_before,
        )
        report["transform_check"] = transform_check
        if not bool(transform_check["ok"]):
            floor_status = "FAIL"
            floor_method = "none"
            fail_reason = "transform_inconsistent"
            fail_reason_detail = (
                f"closure_median_m={transform_check.get('closure_median_m')} "
                f"center_ray_cos={transform_check.get('center_ray_cos')}"
            )
        else:
            rng = np.random.default_rng(scene_seed + 2020)
            ransac_result = estimate_floor_ransac(
                pc=pc,
                p_cam=p_cam_before,
                rng=rng,
                args=args,
                image_w=int(sensor_cfg["width"]),
            )
            ransac_attempts = ransac_result.get("ransac_attempts", [])
            report["ransac_result"] = {
                "ok": bool(ransac_result.get("ok", False)),
                "fail_reason": ransac_result.get("fail_reason"),
                "fail_reason_detail": ransac_result.get("fail_reason_detail"),
                "valid_points_in_roi": ransac_result.get("valid_points_in_roi"),
                "sampled_points": ransac_result.get("sampled_points"),
                "inlier_ratio": ransac_result.get("inlier_ratio"),
                "plane_residual_p50": ransac_result.get("plane_residual_p50"),
                "plane_residual_p95": ransac_result.get("plane_residual_p95"),
                "inlier_coverage": ransac_result.get("inlier_coverage"),
                "ransac_low_height_quantile_used": ransac_result.get("ransac_low_height_quantile_used"),
            }
            if bool(ransac_result.get("ok", False)):
                selected_floor_result = ransac_result
                floor_status = str(ransac_result["floor_status"])
                floor_method = str(ransac_result["floor_method"])
                floor_plane = list(ransac_result["floor_plane"])
                floor_normal = list(ransac_result["floor_normal"])
                floor_height = float(ransac_result["floor_height"])
                cam_height = float(ransac_result["cam_height_above_floor"])
                floor_confidence = float(ransac_result["floor_confidence"])
                step2_ok = True
                fail_reason = None
            else:
                ray_preflight_info = ray_preflight(sim=sim, p_cam=p_cam_before, args=args)
                if bool(ray_preflight_info.get("ok", False)):
                    ray_result = ray_fallback_from_cast(sim=sim, p_cam=p_cam_before, args=args)
                else:
                    ray_result = ray_fallback_depth_proxy(pc=pc, p_cam=p_cam_before, args=args)

                if bool(ray_result.get("ok", False)):
                    selected_floor_result = ray_result
                    floor_status = str(ray_result["floor_status"])
                    floor_method = str(ray_result["floor_method"])
                    floor_plane = list(ray_result["floor_plane"])
                    floor_normal = list(ray_result["floor_normal"])
                    floor_height = float(ray_result["floor_height"])
                    cam_height = float(ray_result["cam_height_above_floor"])
                    floor_confidence = float(ray_result["floor_confidence"])
                    step2_ok = True
                    fail_reason = None
                else:
                    floor_status = "FAIL"
                    floor_method = str(ray_result.get("ray_method") or "none")
                    fail_reason = str(ray_result.get("fail_reason") or ransac_result.get("fail_reason") or "floor_estimation_failed")
                    fail_reason_detail = str(
                        ray_result.get("fail_reason_detail") or ransac_result.get("fail_reason_detail") or "unknown_floor_failure"
                    )

                report["ray_result"] = ray_result

        timings["t_floor_estimate"] = float(time.time() - ts)

        ts = time.time()
        cam_y_before = float(p_cam_before[1])
        cam_y_after = cam_y_before
        floor_pose_apply_applied = False
        floor_pose_apply_reason = None
        obs_after = obs_before

        if step2_ok and floor_height is not None:
            target_cam_y = float(floor_height + float(args.sensor_height_m))
            p_after = p_cam_before.copy()
            p_after[1] = target_cam_y
            obs_after = observe_pose(
                sim=sim,
                agent=agent,
                position=p_after.astype(np.float32),
                yaw_rad=yaw0,
                pitch_rad=pitch0,
            )
            cam_y_after = float(obs_after["pose_readback"]["position"][1])
            floor_pose_apply_applied = True
            floor_pose_apply_reason = "floor_height_applied"
        else:
            floor_pose_apply_reason = "floor_status_not_ok"
            cam_y_after = cam_y_before

        timings["t_apply_pose"] = float(time.time() - ts)

        ts = time.time()
        artifacts = save_step2_render_artifacts(
            scene_dir=scene_dir,
            rgb_before=np.asarray(obs_before["rgb"], dtype=np.uint8),
            depth_before=np.asarray(obs_before["depth"], dtype=np.float32),
            rgb_after=np.asarray(obs_after["rgb"], dtype=np.uint8),
        )
        timings["t_render_after"] = float(time.time() - ts)

        rgb_diff = rgb_l1_diff(np.asarray(obs_before["rgb"], dtype=np.uint8), np.asarray(obs_after["rgb"], dtype=np.uint8))
        depth_shift = depth_shift_stat(
            depth_before=np.asarray(obs_before["depth"], dtype=np.float32),
            depth_after=np.asarray(obs_after["depth"], dtype=np.float32),
            roi_y0_ratio=float(args.roi_y0_ratio),
            roi_y1_ratio=float(args.roi_y1_ratio),
        )
        cam_delta = float(cam_y_after - cam_y_before)
        if abs(cam_delta) > 0.10 and rgb_diff == 0.0:
            warnings.append("FLOOR_APPLY_CHAIN_BROKEN_SUSPECTED")

        if step2_ok:
            status = "OK"
            run_state = "DONE"
        else:
            status = "FAIL"
            run_state = "FAIL"

        sampled_points = int(ransac_result.get("sampled_points", 0)) if isinstance(ransac_result, dict) else 0
        inlier_ratio = selected_floor_result.get("inlier_ratio") if isinstance(selected_floor_result, dict) else None
        residual_p50 = selected_floor_result.get("plane_residual_p50") if isinstance(selected_floor_result, dict) else None
        residual_p95 = selected_floor_result.get("plane_residual_p95") if isinstance(selected_floor_result, dict) else None
        inlier_coverage = selected_floor_result.get("inlier_coverage") if isinstance(selected_floor_result, dict) else None
        ransac_quantile_used = (
            selected_floor_result.get("ransac_low_height_quantile_used") if isinstance(selected_floor_result, dict) else None
        )
        ray_hit_count = selected_floor_result.get("ray_hit_count") if isinstance(selected_floor_result, dict) else None
        ray_hit_median = selected_floor_result.get("ray_hit_height_median") if isinstance(selected_floor_result, dict) else None
        ray_hit_mad = selected_floor_result.get("ray_hit_height_mad") if isinstance(selected_floor_result, dict) else None

        report.update(
            {
                "status": status,
                "run_state": run_state,
                "step2_ok": bool(step2_ok),
                "floor_status": floor_status,
                "floor_method": floor_method,
                "floor_plane": floor_plane,
                "floor_normal": floor_normal,
                "floor_height": floor_height,
                "cam_height_above_floor": cam_height,
                "floor_confidence": float(floor_confidence),
                "valid_points_in_roi": int(pc.get("valid_count", 0)),
                "sampled_points": sampled_points,
                "inlier_ratio": inlier_ratio,
                "plane_residual_p50": residual_p50,
                "plane_residual_p95": residual_p95,
                "inlier_coverage": inlier_coverage,
                "ray_preflight": ray_preflight_info,
                "ray_hit_count": ray_hit_count,
                "ray_hit_height_median": ray_hit_median,
                "ray_hit_height_mad": ray_hit_mad,
                "ransac_attempts": ransac_attempts,
                "ransac_low_height_quantile_used": ransac_quantile_used,
                "floor_plane_raw": floor_plane,
                "floor_plane_smooth": floor_plane,
                "floor_used": "raw",
                "cam_y_before_floor_adjust": cam_y_before,
                "cam_y_after_floor_adjust": cam_y_after,
                "cam_y_delta_floor_adjust": cam_delta,
                "cam_y_minus_floor_height": None if floor_height is None else float(cam_y_after - float(floor_height)),
                "floor_pose_apply_applied": bool(floor_pose_apply_applied),
                "floor_pose_apply_reason": floor_pose_apply_reason,
                "floor_pose_apply_rgb_l1_diff": rgb_diff,
                "depth_after_vs_before_shift": depth_shift,
                "floor_pose_apply_readback_before": obs_before["pose_readback"],
                "floor_pose_apply_readback_after": obs_after["pose_readback"],
                "fail_reason": fail_reason,
                "fail_reason_detail": fail_reason_detail,
                "warnings": warnings,
                "progress_marker": "render_after_done",
                "timeout_stage": None,
            }
        )
    except Exception as exc:
        status = "FAIL"
        run_state = "FAIL"
        step2_ok = False
        floor_status = "FAIL"
        floor_method = "none"
        floor_confidence = 0.0
        fail_reason = "STEP2_EXCEPTION"
        fail_reason_detail = str(exc)
        report.update(
            {
                "status": status,
                "run_state": run_state,
                "step2_ok": False,
                "floor_status": floor_status,
                "floor_method": floor_method,
                "floor_confidence": floor_confidence,
                "fail_reason": fail_reason,
                "fail_reason_detail": fail_reason_detail,
                "warnings": warnings,
                "progress_marker": "exception",
                "timeout_stage": "unknown_or_pre_step2",
            }
        )
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass

    timings["t_write_artifacts"] = float(time.time() - t0 - sum(v for k, v in timings.items() if k != "t_write_artifacts"))
    report["stage_timing_sec"] = timings
    save_json(report_path, report)
    return {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "status": status,
        "run_state": run_state,
        "step2_ok": bool(step2_ok),
        "floor_status": floor_status,
        "floor_method": floor_method,
        "floor_confidence": float(floor_confidence),
        "fail_reason": fail_reason,
        "step2_report_json": str(report_path),
    }


def parse_step2_report_summary(report_path: Path) -> Dict:
    payload = load_json(report_path)
    if not isinstance(payload, dict):
        return {
            "status": "FAIL",
            "run_state": "MISSING",
            "step2_ok": False,
            "floor_status": "FAIL",
            "floor_method": "none",
            "floor_confidence": 0.0,
            "fail_reason": "step2_report_missing_or_invalid",
        }
    return {
        "status": str(payload.get("status") or "FAIL"),
        "run_state": str(payload.get("run_state") or "FAIL"),
        "step2_ok": bool(payload.get("step2_ok", False)),
        "floor_status": str(payload.get("floor_status") or "FAIL"),
        "floor_method": str(payload.get("floor_method") or "none"),
        "floor_confidence": float(payload.get("floor_confidence") or 0.0),
        "fail_reason": payload.get("fail_reason"),
    }


def synthesize_step2_report_for_crash(
    report_path: Path,
    scene_id: str,
    scene_path: str,
    step1_report: Path,
    run_state: str,
    fail_reason: str,
    worker_exit_code: Optional[int],
    log_path: Path,
    env_meta: Dict,
):
    payload = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 2,
        "step_name": "step2_floor_estimation",
        "status": run_state,
        "run_state": run_state,
        "step2_ok": False,
        "floor_status": "FAIL",
        "floor_method": "none",
        "floor_confidence": 0.0,
        "fail_reason": fail_reason,
        "step1_report": str(step1_report),
        "worker_exit_code": worker_exit_code,
        "log_path": str(log_path),
        "environment": env_meta,
    }
    save_json(report_path, payload)


def discover_step1_reports(step1_root: Path, scene_id_filter: Optional[str]) -> List[Path]:
    if not step1_root.exists():
        return []
    rows = []
    for p in sorted(step1_root.glob("*/step1_start_report.json")):
        parent = p.parent.name
        if parent.startswith("_"):
            continue
        if scene_id_filter and parent != scene_id_filter:
            continue
        rows.append(p.resolve())
    return rows


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
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})
    return rows


def build_worker_cmd(script_path: Path, step1_report_path: Path, args) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-step1-report",
        str(step1_report_path),
        "--step1-root",
        str(args.step1_root),
        "--step2-root",
        str(args.step2_root),
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
        "--depth-valid-min-m",
        str(args.depth_valid_min_m),
        "--depth-valid-max-m",
        str(args.depth_valid_max_m),
        "--roi-y0-ratio",
        str(args.roi_y0_ratio),
        "--roi-y1-ratio",
        str(args.roi_y1_ratio),
        "--ransac-low-height-quantiles",
        str(args.ransac_low_height_quantiles),
        "--ransac-max-points",
        str(args.ransac_max_points),
        "--ransac-min-points",
        str(args.ransac_min_points),
        "--ransac-max-iter",
        str(args.ransac_max_iter),
        "--ransac-dist-thresh-m",
        str(args.ransac_dist_thresh_m),
        "--min-inlier-ratio",
        str(args.min_inlier_ratio),
        "--max-residual-m",
        str(args.max_residual_m),
        "--min-inlier-coverage",
        str(args.min_inlier_coverage),
        "--coverage-grid-w",
        str(args.coverage_grid_w),
        "--coverage-grid-h",
        str(args.coverage_grid_h),
        "--min-normal-dot-up",
        str(args.min_normal_dot_up),
        "--min-ny",
        str(args.min_ny),
        "--normal-unit-tol",
        str(args.normal_unit_tol),
        "--cam-height-min-m",
        str(args.cam_height_min_m),
        "--cam-height-max-m",
        str(args.cam_height_max_m),
        "--sensor-height-m",
        str(args.sensor_height_m),
        "--ray-grid",
        str(args.ray_grid),
        "--ray-offset-m",
        str(args.ray_offset_m),
        "--ray-min-normal-dot",
        str(args.ray_min_normal_dot),
        "--ray-min-hits",
        str(args.ray_min_hits),
        "--ray-max-mad-m",
        str(args.ray_max_mad_m),
        "--ray-preflight-min-hit-ratio",
        str(args.ray_preflight_min_hit_ratio),
        "--proxy-radius-m",
        str(args.proxy_radius_m),
        "--proxy-low-quantile",
        str(args.proxy_low_quantile),
        "--lookdown-pitch-deg",
        str(args.lookdown_pitch_deg),
        "--lookdown-delta-y-m",
        str(args.lookdown_delta_y_m),
        "--conf-w1",
        str(args.conf_w1),
        "--conf-w2",
        str(args.conf_w2),
        "--conf-w3",
        str(args.conf_w3),
        "--conf-h-mid-m",
        str(args.conf_h_mid_m),
        "--conf-h-span-m",
        str(args.conf_h_span_m),
        "--no-resume",
    ]
    if args.enforce_depth_valid_max:
        cmd.append("--enforce-depth-valid-max")
    else:
        cmd.append("--no-enforce-depth-valid-max")
    if args.floor_init_lookdown:
        cmd.append("--floor-init-lookdown")
    else:
        cmd.append("--no-floor-init-lookdown")
    if args.disable_physics:
        cmd.append("--disable-physics")
    return cmd


def run_scene_with_subprocess(
    step1_report_path: Path,
    idx: int,
    total: int,
    step2_root: Path,
    env_meta: Dict,
    args,
) -> Dict:
    step1_payload = load_json(step1_report_path) or {}
    scene_id = scene_id_from_step1_report(report_path=step1_report_path, payload=step1_payload)
    scene_path = str(step1_payload.get("scene_path") or "")
    scene_dir = step2_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / "step2_floor_report.json"

    log_dir = step2_root / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scene_id}.log"

    cmd = build_worker_cmd(script_path=Path(__file__).resolve(), step1_report_path=step1_report_path, args=args)
    t0 = time.time()
    worker_exit_code = None
    timed_out = False
    stdout_text = ""
    stderr_text = ""

    try:
        proc = subprocess.run(
            cmd,
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
        worker_exit_code = -9
        stdout_text = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        stderr_text = (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
        stderr_text += f"\nTIMEOUT>{args.worker_timeout_sec}s\n"

    with log_path.open("w", encoding="utf-8") as f:
        if stdout_text:
            f.write(stdout_text)
        if stderr_text:
            if stdout_text and not stdout_text.endswith("\n"):
                f.write("\n")
            f.write(stderr_text)

    elapsed = time.time() - t0
    if timed_out and not report_path.exists():
        synthesize_step2_report_for_crash(
            report_path=report_path,
            scene_id=scene_id,
            scene_path=scene_path,
            step1_report=step1_report_path,
            run_state="TIMEOUT",
            fail_reason="WORKER_TIMEOUT",
            worker_exit_code=worker_exit_code,
            log_path=log_path,
            env_meta=env_meta,
        )
    elif worker_exit_code is not None and worker_exit_code != 0 and not report_path.exists():
        if worker_exit_code < 0:
            run_state = "CRASH_NATIVE"
            fail_reason = f"WORKER_NATIVE_EXIT_{worker_exit_code}"
        else:
            run_state = "WORKER_ERROR"
            fail_reason = f"WORKER_EXIT_NONZERO_{worker_exit_code}"
        synthesize_step2_report_for_crash(
            report_path=report_path,
            scene_id=scene_id,
            scene_path=scene_path,
            step1_report=step1_report_path,
            run_state=run_state,
            fail_reason=fail_reason,
            worker_exit_code=worker_exit_code,
            log_path=log_path,
            env_meta=env_meta,
        )

    parsed = parse_step2_report_summary(report_path)
    return {
        "idx": idx,
        "total": total,
        "scene_id": scene_id,
        "scene_path": scene_path,
        "step1_report": str(step1_report_path),
        "status": parsed["status"],
        "run_state": parsed["run_state"],
        "elapsed_sec": f"{elapsed:.3f}",
        "step2_ok": parsed["step2_ok"],
        "floor_status": parsed["floor_status"],
        "floor_method": parsed["floor_method"],
        "floor_confidence": parsed["floor_confidence"],
        "fail_reason": parsed["fail_reason"],
        "step2_report_json": str(report_path),
        "log_path": str(log_path),
        "worker_exit_code": worker_exit_code,
    }


def run_scene_inline(
    step1_report_path: Path,
    idx: int,
    total: int,
    step2_root: Path,
    env_meta: Dict,
    args,
) -> Dict:
    t0 = time.time()
    result = run_step2_scene_worker(step1_report_path=step1_report_path, step2_root=step2_root, args=args, env_meta=env_meta)
    elapsed = time.time() - t0
    return {
        "idx": idx,
        "total": total,
        "scene_id": result["scene_id"],
        "scene_path": result["scene_path"],
        "step1_report": str(step1_report_path),
        "status": result["status"],
        "run_state": result["run_state"],
        "elapsed_sec": f"{elapsed:.3f}",
        "step2_ok": result["step2_ok"],
        "floor_status": result["floor_status"],
        "floor_method": result["floor_method"],
        "floor_confidence": result["floor_confidence"],
        "fail_reason": result["fail_reason"],
        "step2_report_json": result["step2_report_json"],
        "log_path": "",
        "worker_exit_code": 0,
    }


def run_worker_entry(args):
    env_meta = collect_env_meta()
    step1_report = args.worker_step1_report.resolve()
    result = run_step2_scene_worker(
        step1_report_path=step1_report,
        step2_root=args.step2_root.resolve(),
        args=args,
        env_meta=env_meta,
    )
    print(
        f"[WORKER] {result['scene_id']} | status={result['status']} | run_state={result['run_state']} "
        f"| floor={result['floor_status']}:{result['floor_method']} | conf={result['floor_confidence']:.3f} | fail={result['fail_reason']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1-root", type=Path, default=DEFAULT_STEP1_ROOT)
    parser.add_argument("--step2-root", type=Path, default=None)
    parser.add_argument("--scene-id", type=str, default=None, help="run step2 for one scene_id only (step1/<scene_id>/step1_start_report.json)")
    parser.add_argument("--max-new", type=int, default=0, help="max number of new scenes to process (0: no limit)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--subprocess-isolation", dest="subprocess_isolation", action="store_true", default=True)
    parser.add_argument("--no-subprocess-isolation", dest="subprocess_isolation", action="store_false")
    parser.add_argument("--worker-timeout-sec", type=float, default=360.0)
    parser.add_argument("--worker-step1-report", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--hfov", type=float, default=0.0)
    parser.add_argument("--znear", type=float, default=0.0)
    parser.add_argument("--zfar", type=float, default=0.0)
    parser.add_argument("--disable-physics", action="store_true")

    parser.add_argument("--depth-valid-min-m", type=float, default=1e-4)
    parser.add_argument("--depth-valid-max-m", type=float, default=200.0)
    parser.add_argument("--enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_true", default=False)
    parser.add_argument("--no-enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_false")

    parser.add_argument("--roi-y0-ratio", type=float, default=0.60)
    parser.add_argument("--roi-y1-ratio", type=float, default=0.95)
    parser.add_argument("--ransac-low-height-quantiles", type=str, default="0.20,0.35,0.50,1.00")
    parser.add_argument("--ransac-max-points", type=int, default=12000)
    parser.add_argument("--ransac-min-points", type=int, default=500)
    parser.add_argument("--ransac-max-iter", type=int, default=200)
    parser.add_argument("--ransac-dist-thresh-m", type=float, default=0.03)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.15)
    parser.add_argument("--max-residual-m", type=float, default=0.05)
    parser.add_argument("--min-inlier-coverage", type=float, default=0.15)
    parser.add_argument("--coverage-grid-w", type=int, default=40)
    parser.add_argument("--coverage-grid-h", type=int, default=22)
    parser.add_argument("--min-normal-dot-up", type=float, default=0.90)
    parser.add_argument("--min-ny", type=float, default=0.70)
    parser.add_argument("--normal-unit-tol", type=float, default=1e-3)
    parser.add_argument("--cam-height-min-m", type=float, default=0.8)
    parser.add_argument("--cam-height-max-m", type=float, default=2.2)
    parser.add_argument("--sensor-height-m", type=float, default=1.5)

    parser.add_argument("--ray-grid", type=str, default="3x3", choices=["3x3", "cross5"])
    parser.add_argument("--ray-offset-m", type=float, default=0.15)
    parser.add_argument("--ray-min-normal-dot", type=float, default=0.85)
    parser.add_argument("--ray-min-hits", type=int, default=5)
    parser.add_argument("--ray-max-mad-m", type=float, default=0.05)
    parser.add_argument("--ray-preflight-min-hit-ratio", type=float, default=0.4)
    parser.add_argument("--proxy-radius-m", type=float, default=0.25)
    parser.add_argument("--proxy-low-quantile", type=float, default=0.15)

    parser.add_argument("--floor-init-lookdown", dest="floor_init_lookdown", action="store_true", default=True)
    parser.add_argument("--no-floor-init-lookdown", dest="floor_init_lookdown", action="store_false")
    parser.add_argument("--lookdown-pitch-deg", type=float, default=15.0)
    parser.add_argument("--lookdown-delta-y-m", type=float, default=0.6)

    parser.add_argument("--conf-w1", type=float, default=0.4)
    parser.add_argument("--conf-w2", type=float, default=0.4)
    parser.add_argument("--conf-w3", type=float, default=0.2)
    parser.add_argument("--conf-h-mid-m", type=float, default=1.5)
    parser.add_argument("--conf-h-span-m", type=float, default=0.7)

    args = parser.parse_args()

    args.step1_root = args.step1_root.resolve()
    if args.step2_root is None:
        args.step2_root = args.step1_root.parent / STEP2_DIRNAME
    args.step2_root = args.step2_root.resolve()
    args.step2_root.mkdir(parents=True, exist_ok=True)

    if args.worker_step1_report is not None:
        run_worker_entry(args)
        return

    step1_reports = discover_step1_reports(step1_root=args.step1_root, scene_id_filter=args.scene_id)
    if not step1_reports:
        print(f"No step1_start_report.json found under: {args.step1_root}")
        return

    summary_path = args.step2_root / SUMMARY_PATH_NAME
    existing_rows = normalize_summary_schema(summary_path)
    existing_count = len(existing_rows)

    done_scene_ids = set()
    if args.resume:
        for row in existing_rows:
            sid = (row.get("scene_id") or "").strip()
            if sid:
                done_scene_ids.add(sid)

    write_header = not summary_path.exists() or summary_path.stat().st_size == 0
    env_meta = collect_env_meta()
    processed_new = 0
    inventory_total = len(step1_reports)

    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
            f.flush()

        for loop_i, step1_report_path in enumerate(step1_reports, start=1):
            p1 = load_json(step1_report_path) or {}
            scene_id = scene_id_from_step1_report(report_path=step1_report_path, payload=p1)

            if scene_id in done_scene_ids:
                print(f"[{loop_i}/{inventory_total}] SKIP {scene_id} (already in summary)")
                continue
            if args.max_new > 0 and processed_new >= args.max_new:
                print(f"Reached --max-new={args.max_new}, stop.")
                break

            if args.subprocess_isolation:
                row = run_scene_with_subprocess(
                    step1_report_path=step1_report_path,
                    idx=loop_i,
                    total=inventory_total,
                    step2_root=args.step2_root,
                    env_meta=env_meta,
                    args=args,
                )
            else:
                row = run_scene_inline(
                    step1_report_path=step1_report_path,
                    idx=loop_i,
                    total=inventory_total,
                    step2_root=args.step2_root,
                    env_meta=env_meta,
                    args=args,
                )

            row_id = existing_count + processed_new + 1
            row_out = {
                "row_id": row_id,
                "idx": row["idx"],
                "total": row["total"],
                "scene_id": row["scene_id"],
                "scene_path": row["scene_path"],
                "step1_report": row["step1_report"],
                "status": row["status"],
                "run_state": row["run_state"],
                "elapsed_sec": row["elapsed_sec"],
                "step2_ok": row["step2_ok"],
                "floor_status": row["floor_status"],
                "floor_method": row["floor_method"],
                "floor_confidence": row["floor_confidence"],
                "fail_reason": row["fail_reason"],
                "step2_report_json": row["step2_report_json"],
                "log_path": row["log_path"],
                "worker_exit_code": row["worker_exit_code"],
            }
            writer.writerow(row_out)
            f.flush()

            done_scene_ids.add(scene_id)
            processed_new += 1
            print(
                f"[{row['idx']}/{row['total']}] {scene_id} | {row['status']} | run_state={row['run_state']} "
                f"| floor={row['floor_status']}:{row['floor_method']} | conf={row['floor_confidence']:.3f} "
                f"| t={row['elapsed_sec']}s | fail={row['fail_reason']}"
            )

    print(
        f"Step2 batch done: inventory_total={inventory_total}, newly_processed={processed_new}, "
        f"resume={'on' if args.resume else 'off'}, subprocess_isolation={'on' if args.subprocess_isolation else 'off'}"
    )


if __name__ == "__main__":
    main()
