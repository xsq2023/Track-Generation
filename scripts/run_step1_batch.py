#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from path_defaults import DEFAULT_OUTPUT_ROOT

DEFAULT_STEP0_ROOT = DEFAULT_OUTPUT_ROOT / "step0"
STEP1_DIRNAME = "step1"
SUMMARY_PATH_NAME = "_batch_summary.tsv"
LOG_DIRNAME = "_batch_logs"

SUMMARY_FIELDS = [
    "row_id",
    "idx",
    "total",
    "scene_id",
    "scene_path",
    "step0_scene_init",
    "status",
    "run_state",
    "elapsed_sec",
    "step1_ok",
    "start_mode",
    "navmesh_status",
    "candidates_total",
    "accepted_candidates",
    "best_score",
    "fail_reason",
    "step1_report_json",
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


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=to_json_compatible), encoding="utf-8")


def scene_id_from_path(scene_path: Path) -> str:
    raw = scene_path.stem
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw).strip("_")
    return cleaned or "unknown_scene"


def scene_id_from_scene_init_path(scene_init_path: Path, payload: Optional[Dict]) -> str:
    if isinstance(payload, dict):
        sid = payload.get("scene_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
        sp = payload.get("scene_path")
        if isinstance(sp, str) and sp.strip():
            return scene_id_from_path(Path(sp))
    return scene_init_path.parent.name


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


def resolve_sensor_config(step0_payload: Dict, args) -> Dict:
    base = step0_payload.get("sensor_config") if isinstance(step0_payload.get("sensor_config"), dict) else {}
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


def yaw_to_quat_xyzw(yaw_rad: float) -> np.ndarray:
    q = mn.Quaternion.rotation(mn.Rad(float(yaw_rad)), mn.Vector3(0.0, 1.0, 0.0))
    return np.array([float(q.vector.x), float(q.vector.y), float(q.vector.z), float(q.scalar)], dtype=np.float32)


def forward_from_yaw(yaw_rad: float) -> np.ndarray:
    return np.array([math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype=np.float32)


def set_agent_pose(agent, position: np.ndarray, yaw_rad: float) -> Dict:
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = yaw_to_quat_xyzw(float(yaw_rad))
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


def observe_pose(sim: habitat_sim.Simulator, agent, position: np.ndarray, yaw_rad: float) -> Dict:
    readback = set_agent_pose(agent, position=position, yaw_rad=yaw_rad)
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


def compute_edge_ratio(depth: np.ndarray, valid_mask: np.ndarray, edge_depth_delta_m: float) -> float:
    if depth.ndim != 2:
        return 0.0
    valid_h = valid_mask[:, 1:] & valid_mask[:, :-1]
    valid_v = valid_mask[1:, :] & valid_mask[:-1, :]
    edge_h = np.abs(depth[:, 1:] - depth[:, :-1])
    edge_v = np.abs(depth[1:, :] - depth[:-1, :])
    edge_hits = int(np.sum((edge_h > edge_depth_delta_m) & valid_h) + np.sum((edge_v > edge_depth_delta_m) & valid_v))
    edge_total = int(np.sum(valid_h) + np.sum(valid_v))
    if edge_total <= 0:
        return 0.0
    return float(edge_hits / edge_total)


def compute_depth_metrics(depth: np.ndarray, args) -> Dict:
    total = int(depth.size)
    finite = np.isfinite(depth)
    valid = finite & (depth > float(args.depth_valid_min_m))
    validity_mode = "finite_gt_min"
    if bool(args.enforce_depth_valid_max):
        valid = valid & (depth < float(args.depth_valid_max_m))
        validity_mode = "finite_gt_min_lt_max"
    valid_count = int(np.sum(valid))
    r_valid = float(valid_count / max(total, 1))

    metrics = {
        "r_valid": r_valid,
        "r_near": 0.0,
        "r_far": 1.0,
        "r_edge": 0.0,
        "d05": None,
        "d10": None,
        "depth_iqr": 0.0,
        "depth_p25": None,
        "depth_p75": None,
        "valid_min_m": None,
        "valid_max_m": None,
        "valid_mean_m": None,
        "score": 0.0,
        "score_terms": {},
        "validity_mode": validity_mode,
    }
    if valid_count <= 0:
        metrics["score_terms"] = {
            "w_near_term": 0.0,
            "w_far_term": 0.0,
            "w_edge_term": 0.0,
            "w_iqr_term": 0.0,
            "iqr_norm": 0.0,
        }
        return metrics

    values = depth[valid]
    d05 = float(np.percentile(values, 5.0))
    d10 = float(np.percentile(values, 10.0))
    p25 = float(np.percentile(values, 25.0))
    p75 = float(np.percentile(values, 75.0))
    iqr = float(max(p75 - p25, 0.0))

    r_near = float(np.mean(values < float(args.depth_near_m)))
    r_far = float(np.mean(values > float(args.depth_far_m)))
    r_edge = float(compute_edge_ratio(depth=depth, valid_mask=valid, edge_depth_delta_m=float(args.edge_depth_delta_m)))
    iqr_norm = float(np.clip(iqr / max(float(args.iqr_norm_max_m), 1e-6), 0.0, 1.0))

    w_near_term = float(args.w_near) * r_near
    w_far_term = float(args.w_far) * (1.0 - r_far)
    w_edge_term = float(args.w_edge) * r_edge
    w_iqr_term = float(args.w_iqr) * iqr_norm
    score = float(w_near_term + w_far_term + w_edge_term + w_iqr_term)

    metrics.update(
        {
            "r_near": r_near,
            "r_far": r_far,
            "r_edge": r_edge,
            "d05": d05,
            "d10": d10,
            "depth_iqr": iqr,
            "depth_p25": p25,
            "depth_p75": p75,
            "valid_min_m": float(np.min(values)),
            "valid_max_m": float(np.max(values)),
            "valid_mean_m": float(np.mean(values)),
            "score": score,
            "score_terms": {
                "w_near_term": w_near_term,
                "w_far_term": w_far_term,
                "w_edge_term": w_edge_term,
                "w_iqr_term": w_iqr_term,
                "iqr_norm": iqr_norm,
            },
        }
    )
    return metrics


def reject_reasons_from_metrics(metrics: Dict, args) -> List[str]:
    if float(metrics.get("r_valid", 0.0)) < float(args.min_valid_ratio):
        return ["REJ_INVALID_DEPTH"]

    reasons: List[str] = []
    d05 = metrics.get("d05")
    if d05 is None or float(d05) < float(args.min_d05_m):
        reasons.append("REJ_TOO_CLOSE_WALL")
    r_far = float(metrics.get("r_far", 1.0))
    if r_far > float(args.max_far_ratio):
        reasons.append("REJ_TOO_OPEN")
    # Only enforce texture density after depth validity / wall / openness gates pass.
    if ("REJ_TOO_CLOSE_WALL" not in reasons) and ("REJ_TOO_OPEN" not in reasons):
        if float(metrics.get("r_edge", 0.0)) < float(args.min_edge_ratio):
            reasons.append("REJ_LOW_TEXTURE")
    if float(metrics.get("score", 0.0)) < float(args.score_min):
        reasons.append("REJ_SCORE_LOW")
    return reasons


def gate_pass(metrics: Dict, args) -> bool:
    d05 = metrics.get("d05")
    if d05 is None:
        return False
    return float(metrics.get("r_valid", 0.0)) >= float(args.min_valid_ratio) and float(d05) >= float(args.min_d05_m)


def accept_pose(metrics: Dict, args) -> bool:
    reasons = reject_reasons_from_metrics(metrics=metrics, args=args)
    if reasons:
        return False
    return float(metrics.get("score", 0.0)) >= float(args.score_min)


def evaluate_pose(sim: habitat_sim.Simulator, agent, position: np.ndarray, yaw_rad: float, args) -> Dict:
    obs = observe_pose(sim=sim, agent=agent, position=position, yaw_rad=yaw_rad)
    metrics = compute_depth_metrics(obs["depth"], args=args)
    reasons = reject_reasons_from_metrics(metrics=metrics, args=args)
    metrics["reject_reasons"] = reasons
    metrics["gate_pass"] = gate_pass(metrics=metrics, args=args)
    metrics["score_pass"] = float(metrics["score"]) >= float(args.score_min)
    metrics["accept"] = accept_pose(metrics=metrics, args=args)
    return {
        "position": [float(position[0]), float(position[1]), float(position[2])],
        "yaw_rad": float(yaw_rad),
        "pose_readback": obs["pose_readback"],
        "metrics": metrics,
        "rgb": obs["rgb"],
        "depth": obs["depth"],
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
            "point": np.array([float(hit.point[0]), float(hit.point[1]), float(hit.point[2])], dtype=np.float32),
            "normal": np.array([float(hit.normal[0]), float(hit.normal[1]), float(hit.normal[2])], dtype=np.float32),
            "distance": float(hit.ray_distance),
            "object_id": int(hit.object_id),
        }
    except Exception:
        return None


def no_navmesh_sample_pose(
    sim: habitat_sim.Simulator,
    aabb: Dict,
    rng: np.random.Generator,
    args,
) -> Tuple[Optional[Dict], Optional[str]]:
    min_v = np.asarray(aabb["min"], dtype=np.float32)
    max_v = np.asarray(aabb["max"], dtype=np.float32)
    x = float(rng.uniform(min_v[0], max_v[0]))
    z = float(rng.uniform(min_v[2], max_v[2]))
    top_y = float(max_v[1] + float(args.raycast_top_pad_m))

    down_hit = first_hit(
        sim=sim,
        origin=np.array([x, top_y, z], dtype=np.float32),
        direction=np.array([0.0, -1.0, 0.0], dtype=np.float32),
    )
    if down_hit is None:
        if bool(args.allow_floor_proxy):
            cam_pos = np.array([x, float(min_v[1] + float(args.cam_height_m)), z], dtype=np.float32)
            floor_source = "aabb_min_proxy"
        else:
            return None, "REJ_NO_FLOOR_HIT"
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        floor_dot = float(np.dot(down_hit["normal"], up))
        if floor_dot < float(args.min_floor_normal_dot):
            if bool(args.allow_floor_proxy):
                cam_pos = np.array([x, float(min_v[1] + float(args.cam_height_m)), z], dtype=np.float32)
                floor_source = "aabb_min_proxy"
            else:
                return None, "REJ_NO_FLOOR_HIT"
        else:
            cam_pos = down_hit["point"] + np.array([0.0, float(args.cam_height_m), 0.0], dtype=np.float32)
            floor_source = "raycast_hit"

    directions = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    for d in directions:
        hit = first_hit(sim=sim, origin=cam_pos, direction=d)
        if hit is not None and hit["distance"] < float(args.clearance_min_m):
            return None, "REJ_CLEARANCE_FAIL"

    yaw_trials = int(max(args.forward_yaw_trials, 1)) if bool(args.forward_hit_gate) else 1
    forward_hit_max_m = float(args.forward_hit_max_m)
    if bool(args.adaptive_forward_hit_max):
        scene_max_dim = float(aabb.get("max_dim") or 0.0)
        adaptive_m = min(float(args.forward_hit_max_cap_m), scene_max_dim * float(args.forward_hit_max_scale))
        forward_hit_max_m = max(forward_hit_max_m, adaptive_m)
    saw_hit_too_far = False
    for _ in range(yaw_trials):
        yaw = float(rng.uniform(-math.pi, math.pi))
        if not bool(args.forward_hit_gate):
            return {"position": cam_pos, "yaw_rad": yaw, "floor_source": floor_source}, None

        forward_dir = forward_from_yaw(yaw)
        fwd_hit = first_hit(sim=sim, origin=cam_pos, direction=forward_dir)
        if fwd_hit is None:
            continue
        if float(fwd_hit["distance"]) <= forward_hit_max_m:
            return {"position": cam_pos, "yaw_rad": yaw, "floor_source": floor_source}, None
        saw_hit_too_far = True

    if saw_hit_too_far:
        return None, "REJ_FORWARD_TOO_FAR"
    return None, "REJ_FORWARD_MISS"


def navmesh_sample_pose(pathfinder, rng: np.random.Generator, cam_height_m: float) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        p = pathfinder.get_random_navigable_point()
    except Exception:
        return None, "REJ_NO_NAV_POINT"
    arr = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        return None, "REJ_NO_NAV_POINT"
    yaw = float(rng.uniform(-math.pi, math.pi))
    arr[1] += float(cam_height_m)
    return {"position": arr, "yaw_rad": yaw}, None


def sample_start_candidates(
    sim: habitat_sim.Simulator,
    agent,
    use_navmesh: bool,
    aabb: Dict,
    rng: np.random.Generator,
    args,
) -> Dict:
    accepted = []
    reject_hist = Counter()

    for _ in range(int(args.max_start_samples)):
        if use_navmesh:
            pose, early_reject = navmesh_sample_pose(sim.pathfinder, rng=rng, cam_height_m=float(args.cam_height_m))
        else:
            pose, early_reject = no_navmesh_sample_pose(sim=sim, aabb=aabb, rng=rng, args=args)

        if early_reject is not None:
            reject_hist[early_reject] += 1
            continue
        if pose is None:
            reject_hist["REJ_START_POSE_INVALID"] += 1
            continue

        eval_result = evaluate_pose(
            sim=sim,
            agent=agent,
            position=np.asarray(pose["position"], dtype=np.float32),
            yaw_rad=float(pose["yaw_rad"]),
            args=args,
        )
        reasons = list(eval_result["metrics"]["reject_reasons"])
        if reasons:
            for reason in reasons:
                reject_hist[reason] += 1
            continue

        accepted.append(
            {
                "pose": {
                    "position": eval_result["position"],
                    "yaw_rad": eval_result["yaw_rad"],
                },
                "metrics": eval_result["metrics"],
                "score": float(eval_result["metrics"]["score"]),
                "start_source": "navmesh" if use_navmesh else "no_navmesh_sampling",
            }
        )

    accepted.sort(key=lambda x: x["score"], reverse=True)
    top_k = max(int(args.top_k), 1)
    return {
        "start_candidates_total": int(args.max_start_samples),
        "accepted_candidates": int(len(accepted)),
        "top_candidates": accepted[:top_k],
        "reject_histogram": dict(reject_hist),
    }


def apply_action(
    pose: Dict,
    action_name: str,
    pathfinder,
    args,
) -> Dict:
    pos = np.asarray(pose["position"], dtype=np.float32)
    yaw = float(pose["yaw_rad"])

    if action_name == "turn_left_small":
        return {"position": pos, "yaw_rad": yaw + math.radians(float(args.turn_small_deg))}
    if action_name == "turn_right_small":
        return {"position": pos, "yaw_rad": yaw - math.radians(float(args.turn_small_deg))}

    if action_name == "forward_small":
        delta = float(args.forward_small_m)
    elif action_name == "forward_medium":
        delta = float(args.forward_medium_m)
    elif action_name == "back_small":
        delta = -float(args.back_small_m)
    else:
        delta = 0.0

    target = pos + forward_from_yaw(yaw) * delta
    if pathfinder is not None and bool(pathfinder.is_loaded):
        try:
            stepped = pathfinder.try_step(pos, target)
            stepped_np = np.array([float(stepped[0]), float(stepped[1]), float(stepped[2])], dtype=np.float32)
            if np.all(np.isfinite(stepped_np)):
                target = stepped_np
        except Exception:
            pass
    return {"position": target, "yaw_rad": yaw}


def run_lookahead_search(
    sim: habitat_sim.Simulator,
    agent,
    start_pose: Dict,
    pathfinder,
    args,
) -> Dict:
    actions = ["turn_left_small", "turn_right_small", "forward_small", "forward_medium", "back_small"]
    current_pose = {
        "position": np.asarray(start_pose["position"], dtype=np.float32),
        "yaw_rad": float(start_pose["yaw_rad"]),
    }
    current_eval = evaluate_pose(
        sim=sim,
        agent=agent,
        position=current_pose["position"],
        yaw_rad=current_pose["yaw_rad"],
        args=args,
    )
    best_eval = current_eval
    trace = []

    if bool(current_eval["metrics"]["accept"]):
        return {"success": True, "best_eval": best_eval, "trace": trace}

    for step_idx in range(int(args.max_actions_per_start)):
        trial_rows = []
        for action_name in actions:
            next_pose = apply_action(pose=current_pose, action_name=action_name, pathfinder=pathfinder, args=args)
            eval_result = evaluate_pose(
                sim=sim,
                agent=agent,
                position=np.asarray(next_pose["position"], dtype=np.float32),
                yaw_rad=float(next_pose["yaw_rad"]),
                args=args,
            )
            trial_rows.append(
                {
                    "action": action_name,
                    "pose": next_pose,
                    "eval": eval_result,
                }
            )

        if not trial_rows:
            break
        trial_rows.sort(key=lambda x: float(x["eval"]["metrics"]["score"]), reverse=True)
        chosen = trial_rows[0]
        current_pose = {
            "position": np.asarray(chosen["pose"]["position"], dtype=np.float32),
            "yaw_rad": float(chosen["pose"]["yaw_rad"]),
        }
        current_eval = chosen["eval"]
        if float(current_eval["metrics"]["score"]) >= float(best_eval["metrics"]["score"]):
            best_eval = current_eval

        trace.append(
            {
                "step": int(step_idx),
                "action": chosen["action"],
                "score": float(current_eval["metrics"]["score"]),
                "accept": bool(current_eval["metrics"]["accept"]),
                "gate_pass": bool(current_eval["metrics"]["gate_pass"]),
                "r_valid": current_eval["metrics"]["r_valid"],
                "r_near": current_eval["metrics"]["r_near"],
                "r_far": current_eval["metrics"]["r_far"],
                "r_edge": current_eval["metrics"]["r_edge"],
                "d05": current_eval["metrics"]["d05"],
                "depth_iqr": current_eval["metrics"]["depth_iqr"],
            }
        )

        if bool(current_eval["metrics"]["accept"]):
            return {"success": True, "best_eval": best_eval, "trace": trace}

    return {"success": bool(best_eval["metrics"]["accept"]), "best_eval": best_eval, "trace": trace}


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


def save_step1_artifacts(scene_dir: Path, eval_result: Dict, depth_clip_max_m: float) -> Dict:
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(eval_result["rgb"], dtype=np.uint8)
    depth = np.asarray(eval_result["depth"], dtype=np.float32)

    rgb_path = scene_dir / "step1_rgb.png"
    depth_path = scene_dir / "step1_depth.png"
    depth_clip_path = scene_dir / "step1_depth_clip.png"
    depth_npy_path = scene_dir / "step1_depth.npy"

    Image.fromarray(rgb).save(rgb_path)
    Image.fromarray(normalize_depth_vis(depth, depth_clip_max_m=depth_clip_max_m)).save(depth_path)
    depth_clip = np.clip(np.where(np.isfinite(depth), depth, 0.0), 0.0, depth_clip_max_m)
    depth_clip_img = np.round(depth_clip / max(depth_clip_max_m, 1e-6) * 255.0).astype(np.uint8)
    Image.fromarray(depth_clip_img).save(depth_clip_path)
    np.save(depth_npy_path, depth)

    return {
        "step1_rgb": str(rgb_path),
        "step1_depth": str(depth_path),
        "step1_depth_clip": str(depth_clip_path),
        "step1_depth_npy": str(depth_npy_path),
    }


def stable_scene_seed(base_seed: int, scene_id: str) -> int:
    digest = hashlib.sha1(scene_id.encode("utf-8")).hexdigest()[:8]
    value = int(digest, 16)
    return int((int(base_seed) + value) % (2**31 - 1))


def summarize_fail_reason(reject_hist: Dict[str, int], default_reason: str) -> str:
    if not reject_hist:
        return default_reason
    key = sorted(reject_hist.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return str(key)


def try_load_navmesh_from_step0(sim: habitat_sim.Simulator, step0_payload: Dict) -> Dict:
    navmesh_status = str(step0_payload.get("navmesh_status") or "UNKNOWN")
    cache_path_raw = step0_payload.get("navmesh_cache_path")
    if navmesh_status not in {"OK", "RECOMPUTED_OK"}:
        return {
            "loaded": False,
            "source": "step0_status_gate",
            "cache_path": None,
            "error": f"step0_navmesh_status_{navmesh_status}",
        }
    if not isinstance(cache_path_raw, str) or not cache_path_raw.strip():
        return {
            "loaded": False,
            "source": "cache_missing",
            "cache_path": None,
            "error": "step0_cache_path_missing",
        }

    cache_path = Path(cache_path_raw)
    if not cache_path.exists():
        return {
            "loaded": False,
            "source": "cache_missing",
            "cache_path": str(cache_path),
            "error": "step0_cache_path_not_found",
        }

    try:
        loaded = bool(sim.pathfinder.load_nav_mesh(str(cache_path)))
        if not loaded or not bool(sim.pathfinder.is_loaded):
            return {
                "loaded": False,
                "source": "cache_load",
                "cache_path": str(cache_path),
                "error": "load_nav_mesh_false",
            }
        p = sim.pathfinder.get_random_navigable_point()
        arr = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            return {
                "loaded": False,
                "source": "cache_load",
                "cache_path": str(cache_path),
                "error": "random_navigable_nonfinite",
            }
        return {
            "loaded": True,
            "source": "step0_cache",
            "cache_path": str(cache_path),
            "error": None,
        }
    except Exception as exc:
        return {
            "loaded": False,
            "source": "cache_load_exception",
            "cache_path": str(cache_path),
            "error": f"{exc}",
        }


def dependency_gate(step0_payload: Dict) -> Tuple[bool, str]:
    if not bool(step0_payload.get("scene_init_ok", False)):
        return False, "STEP0_NOT_READY"
    if bool(step0_payload.get("quarantine", False)):
        return False, "STEP0_QUARANTINED"
    return True, "OK"


def run_step1_scene_worker(scene_init_path: Path, step1_root: Path, args, env_meta: Dict) -> Dict:
    step0_payload = load_json(scene_init_path)
    scene_id = scene_id_from_scene_init_path(scene_init_path=scene_init_path, payload=step0_payload)
    scene_dir = step1_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / "step1_start_report.json"

    if not isinstance(step0_payload, dict):
        payload = {
            "scene_id": scene_id,
            "scene_path": None,
            "pipeline_stage": 1,
            "step_name": "step1_indoor_start",
            "status": "FAIL",
            "run_state": "FAIL",
            "step1_ok": False,
            "fail_reason": "STEP0_SCENE_INIT_MISSING_OR_INVALID",
            "step0_scene_init": str(scene_init_path),
            "environment": env_meta,
            "artifacts": {
                "step1_report_json": str(report_path),
            },
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": "",
            "status": "FAIL",
            "run_state": "FAIL",
            "step1_ok": False,
            "start_mode": "none",
            "navmesh_status": "UNKNOWN",
            "candidates_total": 0,
            "accepted_candidates": 0,
            "best_score": None,
            "fail_reason": "STEP0_SCENE_INIT_MISSING_OR_INVALID",
            "step1_report_json": str(report_path),
        }

    scene_path = str(step0_payload.get("scene_path") or "")
    dep_ok, dep_reason = dependency_gate(step0_payload)
    if not dep_ok:
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 1,
            "step_name": "step1_indoor_start",
            "status": "SKIP",
            "run_state": "SKIP",
            "step1_ok": False,
            "fail_reason": dep_reason,
            "step0_scene_init": str(scene_init_path),
            "step0_dependency": {
                "scene_init_ok": bool(step0_payload.get("scene_init_ok", False)),
                "quarantine": bool(step0_payload.get("quarantine", False)),
                "navmesh_status": step0_payload.get("navmesh_status"),
                "fallback_plan": step0_payload.get("fallback_plan"),
            },
            "environment": env_meta,
            "artifacts": {
                "step1_report_json": str(report_path),
            },
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "SKIP",
            "run_state": "SKIP",
            "step1_ok": False,
            "start_mode": "none",
            "navmesh_status": str(step0_payload.get("navmesh_status") or "UNKNOWN"),
            "candidates_total": 0,
            "accepted_candidates": 0,
            "best_score": None,
            "fail_reason": dep_reason,
            "step1_report_json": str(report_path),
        }

    scene_source_raw = step0_payload.get("post_scene_source") or step0_payload.get("scene_path")
    scene_source = Path(scene_source_raw) if isinstance(scene_source_raw, str) else None
    if scene_source is None or not scene_source.exists():
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 1,
            "step_name": "step1_indoor_start",
            "status": "FAIL",
            "run_state": "FAIL",
            "step1_ok": False,
            "fail_reason": "STEP0_SCENE_SOURCE_MISSING",
            "step0_scene_init": str(scene_init_path),
            "scene_source": str(scene_source_raw),
            "environment": env_meta,
            "artifacts": {
                "step1_report_json": str(report_path),
            },
        }
        save_json(report_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step1_ok": False,
            "start_mode": "none",
            "navmesh_status": str(step0_payload.get("navmesh_status") or "UNKNOWN"),
            "candidates_total": 0,
            "accepted_candidates": 0,
            "best_score": None,
            "fail_reason": "STEP0_SCENE_SOURCE_MISSING",
            "step1_report_json": str(report_path),
        }

    sensor_cfg = resolve_sensor_config(step0_payload=step0_payload, args=args)
    sim = None
    report_payload = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 1,
        "step_name": "step1_indoor_start",
        "step0_scene_init": str(scene_init_path),
        "scene_source": str(scene_source),
        "step0_dependency": {
            "scene_init_ok": bool(step0_payload.get("scene_init_ok", False)),
            "quarantine": bool(step0_payload.get("quarantine", False)),
            "navmesh_status": step0_payload.get("navmesh_status"),
            "fallback_plan": step0_payload.get("fallback_plan"),
        },
        "sensor_config": sensor_cfg,
        "thresholds": {
            "depth_valid_min_m": float(args.depth_valid_min_m),
            "depth_valid_max_m": float(args.depth_valid_max_m),
            "enforce_depth_valid_max": bool(args.enforce_depth_valid_max),
            "depth_validity_mode": "finite_gt_min_lt_max" if bool(args.enforce_depth_valid_max) else "finite_gt_min",
            "depth_near_m": float(args.depth_near_m),
            "depth_far_m": float(args.depth_far_m),
            "min_valid_ratio": float(args.min_valid_ratio),
            "min_d05_m": float(args.min_d05_m),
            "max_far_ratio": float(args.max_far_ratio),
            "min_edge_ratio": float(args.min_edge_ratio),
            "edge_depth_delta_m": float(args.edge_depth_delta_m),
            "iqr_norm_max_m": float(args.iqr_norm_max_m),
            "score_min": float(args.score_min),
        },
        "score_weights": {
            "w_near": float(args.w_near),
            "w_far": float(args.w_far),
            "w_edge": float(args.w_edge),
            "w_iqr": float(args.w_iqr),
        },
        "search_params": {
            "max_start_samples": int(args.max_start_samples),
            "top_k": int(args.top_k),
            "max_actions_per_start": int(args.max_actions_per_start),
            "turn_small_deg": float(args.turn_small_deg),
            "forward_small_m": float(args.forward_small_m),
            "forward_medium_m": float(args.forward_medium_m),
            "back_small_m": float(args.back_small_m),
        },
        "no_navmesh_params": {
            "cam_height_m": float(args.cam_height_m),
            "raycast_top_pad_m": float(args.raycast_top_pad_m),
            "min_floor_normal_dot": float(args.min_floor_normal_dot),
            "clearance_min_m": float(args.clearance_min_m),
            "allow_floor_proxy": bool(args.allow_floor_proxy),
            "forward_hit_gate": bool(args.forward_hit_gate),
            "forward_hit_max_m": float(args.forward_hit_max_m),
            "forward_yaw_trials": int(args.forward_yaw_trials),
            "adaptive_forward_hit_max": bool(args.adaptive_forward_hit_max),
            "forward_hit_max_scale": float(args.forward_hit_max_scale),
            "forward_hit_max_cap_m": float(args.forward_hit_max_cap_m),
        },
        "environment": env_meta,
    }

    status = "FAIL"
    run_state = "FAIL"
    step1_ok = False
    fail_reason = "UNKNOWN"
    start_mode = "none"
    navmesh_probe = {}
    scene_aabb = None
    start_probe = {}
    best_eval = None
    chosen_pose = None
    planner_trace = []
    artifacts = {}

    try:
        scene_seed = stable_scene_seed(base_seed=int(args.seed), scene_id=scene_id)
        rng = np.random.default_rng(scene_seed)
        sim = build_sim(
            scene_source=scene_source,
            sensor_cfg=sensor_cfg,
            seed=scene_seed,
            enable_physics=(not args.disable_physics),
        )
        agent = sim.initialize_agent(0)
        scene_aabb = compute_scene_aabb(sim)
        navmesh_probe = try_load_navmesh_from_step0(sim=sim, step0_payload=step0_payload)
        use_navmesh = bool(navmesh_probe.get("loaded", False))
        start_mode = "navmesh" if use_navmesh else "no_navmesh_sampling"

        start_probe = sample_start_candidates(
            sim=sim,
            agent=agent,
            use_navmesh=use_navmesh,
            aabb=scene_aabb,
            rng=rng,
            args=args,
        )
        top_candidates = list(start_probe.get("top_candidates", []))

        if not top_candidates:
            fail_reason = summarize_fail_reason(
                reject_hist=start_probe.get("reject_histogram", {}),
                default_reason="NO_ACCEPTED_START_CANDIDATE",
            )
        else:
            best_search = None
            for candidate in top_candidates:
                search = run_lookahead_search(
                    sim=sim,
                    agent=agent,
                    start_pose=candidate["pose"],
                    pathfinder=sim.pathfinder if use_navmesh else None,
                    args=args,
                )
                if best_search is None or float(search["best_eval"]["metrics"]["score"]) > float(
                    best_search["best_eval"]["metrics"]["score"]
                ):
                    best_search = search
                if bool(search["success"]):
                    break

            if best_search is not None:
                best_eval = best_search["best_eval"]
                planner_trace = best_search["trace"]
                chosen_pose = {
                    "position": best_eval["position"],
                    "yaw_rad": best_eval["yaw_rad"],
                    "pose_readback": best_eval["pose_readback"],
                }
                artifacts = save_step1_artifacts(
                    scene_dir=scene_dir,
                    eval_result=best_eval,
                    depth_clip_max_m=float(args.depth_clip_max_m),
                )
                step1_ok = bool(best_search["success"])
                if not step1_ok and bool(best_eval["metrics"]["accept"]):
                    step1_ok = True
                status = "OK" if step1_ok else "FAIL"
                run_state = "OK" if step1_ok else "FAIL"
                fail_reason = None if step1_ok else "LOOKAHEAD_SCORE_NOT_REACHED"
            else:
                fail_reason = "LOOKAHEAD_INTERNAL_EMPTY"
    except Exception as exc:
        status = "FAIL"
        run_state = "FAIL"
        step1_ok = False
        fail_reason = f"STEP1_EXCEPTION:{exc}"
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass

    if status != "OK":
        run_state = "FAIL" if run_state == "OK" else run_state

    report_payload.update(
        {
            "status": status,
            "run_state": run_state,
            "step1_ok": bool(step1_ok),
            "start_mode": start_mode,
            "fail_reason": fail_reason,
            "navmesh_probe": navmesh_probe,
            "scene_aabb": scene_aabb,
            "start_candidates_total": int(start_probe.get("start_candidates_total", 0)),
            "accepted_candidates": int(start_probe.get("accepted_candidates", 0)),
            "best_score": float(best_eval["metrics"]["score"]) if isinstance(best_eval, dict) else None,
            "best_metrics": best_eval["metrics"] if isinstance(best_eval, dict) else None,
            "chosen_start_pose": chosen_pose,
            "planner_trace": planner_trace,
            "reject_histogram": start_probe.get("reject_histogram", {}),
            "artifacts": {
                "step1_report_json": str(report_path),
                **artifacts,
            },
        }
    )
    save_json(report_path, report_payload)

    return {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "status": status,
        "run_state": run_state,
        "step1_ok": bool(step1_ok),
        "start_mode": start_mode,
        "navmesh_status": str(step0_payload.get("navmesh_status") or "UNKNOWN"),
        "candidates_total": int(start_probe.get("start_candidates_total", 0)),
        "accepted_candidates": int(start_probe.get("accepted_candidates", 0)),
        "best_score": float(best_eval["metrics"]["score"]) if isinstance(best_eval, dict) else None,
        "fail_reason": fail_reason,
        "step1_report_json": str(report_path),
    }


def parse_step1_report_summary(report_path: Path) -> Dict:
    payload = load_json(report_path)
    if not isinstance(payload, dict):
        return {
            "status": "FAIL",
            "run_state": "MISSING",
            "step1_ok": False,
            "start_mode": "none",
            "candidates_total": 0,
            "accepted_candidates": 0,
            "best_score": None,
            "fail_reason": "step1_report_missing_or_invalid",
        }
    return {
        "status": str(payload.get("status") or "FAIL"),
        "run_state": str(payload.get("run_state") or "FAIL"),
        "step1_ok": bool(payload.get("step1_ok", False)),
        "start_mode": str(payload.get("start_mode") or "none"),
        "candidates_total": int(payload.get("start_candidates_total") or 0),
        "accepted_candidates": int(payload.get("accepted_candidates") or 0),
        "best_score": payload.get("best_score"),
        "fail_reason": payload.get("fail_reason"),
    }


def synthesize_step1_report_for_crash(
    report_path: Path,
    scene_id: str,
    scene_path: str,
    scene_init_path: Path,
    run_state: str,
    fail_reason: str,
    worker_exit_code: Optional[int],
    log_path: Path,
    env_meta: Dict,
):
    payload = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 1,
        "step_name": "step1_indoor_start",
        "status": run_state,
        "run_state": run_state,
        "step1_ok": False,
        "fail_reason": fail_reason,
        "step0_scene_init": str(scene_init_path),
        "worker_exit_code": worker_exit_code,
        "log_path": str(log_path),
        "environment": env_meta,
        "artifacts": {
            "step1_report_json": str(report_path),
        },
    }
    save_json(report_path, payload)


def discover_scene_init_paths(step0_root: Path, scene_id_filter: Optional[str]) -> List[Path]:
    if not step0_root.exists():
        return []
    rows = []
    for p in sorted(step0_root.glob("*/scene_init.json")):
        parent_name = p.parent.name
        if parent_name.startswith("_"):
            continue
        if scene_id_filter and parent_name != scene_id_filter:
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


def build_worker_cmd(script_path: Path, scene_init_path: Path, args) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-scene-init",
        str(scene_init_path),
        "--step0-root",
        str(args.step0_root),
        "--step1-root",
        str(args.step1_root),
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
        "--depth-valid-min-m",
        str(args.depth_valid_min_m),
        "--depth-valid-max-m",
        str(args.depth_valid_max_m),
        "--depth-near-m",
        str(args.depth_near_m),
        "--depth-far-m",
        str(args.depth_far_m),
        "--min-valid-ratio",
        str(args.min_valid_ratio),
        "--min-d05-m",
        str(args.min_d05_m),
        "--max-far-ratio",
        str(args.max_far_ratio),
        "--min-edge-ratio",
        str(args.min_edge_ratio),
        "--edge-depth-delta-m",
        str(args.edge_depth_delta_m),
        "--iqr-norm-max-m",
        str(args.iqr_norm_max_m),
        "--w-near",
        str(args.w_near),
        "--w-far",
        str(args.w_far),
        "--w-edge",
        str(args.w_edge),
        "--w-iqr",
        str(args.w_iqr),
        "--score-min",
        str(args.score_min),
        "--max-start-samples",
        str(args.max_start_samples),
        "--top-k",
        str(args.top_k),
        "--max-actions-per-start",
        str(args.max_actions_per_start),
        "--turn-small-deg",
        str(args.turn_small_deg),
        "--forward-small-m",
        str(args.forward_small_m),
        "--forward-medium-m",
        str(args.forward_medium_m),
        "--back-small-m",
        str(args.back_small_m),
        "--cam-height-m",
        str(args.cam_height_m),
        "--raycast-top-pad-m",
        str(args.raycast_top_pad_m),
        "--min-floor-normal-dot",
        str(args.min_floor_normal_dot),
        "--clearance-min-m",
        str(args.clearance_min_m),
        "--no-resume",
    ]
    if args.enforce_depth_valid_max:
        cmd.append("--enforce-depth-valid-max")
    else:
        cmd.append("--no-enforce-depth-valid-max")
    if args.disable_physics:
        cmd.append("--disable-physics")
    if args.allow_floor_proxy:
        cmd.append("--allow-floor-proxy")
    else:
        cmd.append("--no-allow-floor-proxy")
    if args.forward_hit_gate:
        cmd.append("--forward-hit-gate")
    else:
        cmd.append("--no-forward-hit-gate")
    cmd.extend(
        [
            "--forward-hit-max-m",
            str(args.forward_hit_max_m),
            "--forward-yaw-trials",
            str(args.forward_yaw_trials),
            "--forward-hit-max-scale",
            str(args.forward_hit_max_scale),
            "--forward-hit-max-cap-m",
            str(args.forward_hit_max_cap_m),
        ]
    )
    if args.adaptive_forward_hit_max:
        cmd.append("--adaptive-forward-hit-max")
    else:
        cmd.append("--no-adaptive-forward-hit-max")
    return cmd


def run_scene_with_subprocess(
    scene_init_path: Path,
    idx: int,
    total: int,
    step1_root: Path,
    env_meta: Dict,
    args,
) -> Dict:
    step0_payload = load_json(scene_init_path) or {}
    scene_id = scene_id_from_scene_init_path(scene_init_path=scene_init_path, payload=step0_payload)
    scene_path = str(step0_payload.get("scene_path") or "")

    scene_dir = step1_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / "step1_start_report.json"

    log_dir = step1_root / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scene_id}.log"

    cmd = build_worker_cmd(script_path=Path(__file__).resolve(), scene_init_path=scene_init_path, args=args)
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
        synthesize_step1_report_for_crash(
            report_path=report_path,
            scene_id=scene_id,
            scene_path=scene_path,
            scene_init_path=scene_init_path,
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
        synthesize_step1_report_for_crash(
            report_path=report_path,
            scene_id=scene_id,
            scene_path=scene_path,
            scene_init_path=scene_init_path,
            run_state=run_state,
            fail_reason=fail_reason,
            worker_exit_code=worker_exit_code,
            log_path=log_path,
            env_meta=env_meta,
        )

    parsed = parse_step1_report_summary(report_path)
    return {
        "idx": idx,
        "total": total,
        "scene_id": scene_id,
        "scene_path": scene_path,
        "step0_scene_init": str(scene_init_path),
        "status": parsed["status"],
        "run_state": parsed["run_state"],
        "elapsed_sec": f"{elapsed:.3f}",
        "step1_ok": parsed["step1_ok"],
        "start_mode": parsed["start_mode"],
        "navmesh_status": str(step0_payload.get("navmesh_status") or "UNKNOWN"),
        "candidates_total": parsed["candidates_total"],
        "accepted_candidates": parsed["accepted_candidates"],
        "best_score": parsed["best_score"],
        "fail_reason": parsed["fail_reason"],
        "step1_report_json": str(report_path),
        "log_path": str(log_path),
        "worker_exit_code": worker_exit_code,
    }


def run_scene_inline(
    scene_init_path: Path,
    idx: int,
    total: int,
    step1_root: Path,
    env_meta: Dict,
    args,
) -> Dict:
    t0 = time.time()
    result = run_step1_scene_worker(scene_init_path=scene_init_path, step1_root=step1_root, args=args, env_meta=env_meta)
    elapsed = time.time() - t0
    return {
        "idx": idx,
        "total": total,
        "scene_id": result["scene_id"],
        "scene_path": result["scene_path"],
        "step0_scene_init": str(scene_init_path),
        "status": result["status"],
        "run_state": result["run_state"],
        "elapsed_sec": f"{elapsed:.3f}",
        "step1_ok": result["step1_ok"],
        "start_mode": result["start_mode"],
        "navmesh_status": result["navmesh_status"],
        "candidates_total": result["candidates_total"],
        "accepted_candidates": result["accepted_candidates"],
        "best_score": result["best_score"],
        "fail_reason": result["fail_reason"],
        "step1_report_json": result["step1_report_json"],
        "log_path": "",
        "worker_exit_code": 0,
    }


def run_worker_entry(args):
    env_meta = collect_env_meta()
    scene_init_path = args.worker_scene_init.resolve()
    result = run_step1_scene_worker(
        scene_init_path=scene_init_path,
        step1_root=args.step1_root.resolve(),
        args=args,
        env_meta=env_meta,
    )
    print(
        f"[WORKER] {result['scene_id']} | status={result['status']} | run_state={result['run_state']} "
        f"| mode={result['start_mode']} | best={result['best_score']} | fail={result['fail_reason']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step0-root", type=Path, default=DEFAULT_STEP0_ROOT)
    parser.add_argument("--step1-root", type=Path, default=None)
    parser.add_argument("--scene-id", type=str, default=None, help="run step1 for one scene_id only (step0/<scene_id>/scene_init.json)")
    parser.add_argument("--max-new", type=int, default=0, help="max number of new scenes to process (0: no limit)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--subprocess-isolation", dest="subprocess_isolation", action="store_true", default=True)
    parser.add_argument("--no-subprocess-isolation", dest="subprocess_isolation", action="store_false")
    parser.add_argument("--worker-timeout-sec", type=float, default=360.0)
    parser.add_argument("--worker-scene-init", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=0, help="override width; <=0 means inherit step0 sensor_config")
    parser.add_argument("--height", type=int, default=0, help="override height; <=0 means inherit step0 sensor_config")
    parser.add_argument("--hfov", type=float, default=0.0, help="override hfov; <=0 means inherit step0 sensor_config")
    parser.add_argument("--znear", type=float, default=0.0)
    parser.add_argument("--zfar", type=float, default=0.0)
    parser.add_argument("--depth-clip-max-m", type=float, default=10.0)
    parser.add_argument("--disable-physics", action="store_true")

    parser.add_argument("--depth-valid-min-m", type=float, default=1e-4)
    parser.add_argument("--depth-valid-max-m", type=float, default=200.0)
    parser.add_argument("--enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_true", default=False)
    parser.add_argument("--no-enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_false")
    parser.add_argument("--depth-near-m", type=float, default=2.0)
    parser.add_argument("--depth-far-m", type=float, default=8.0)
    parser.add_argument("--min-valid-ratio", type=float, default=0.6)
    parser.add_argument("--min-d05-m", type=float, default=0.2)
    parser.add_argument("--max-far-ratio", type=float, default=0.6)
    parser.add_argument("--min-edge-ratio", type=float, default=0.01)
    parser.add_argument("--edge-depth-delta-m", type=float, default=0.03)
    parser.add_argument("--iqr-norm-max-m", type=float, default=3.0)

    parser.add_argument("--w-near", type=float, default=0.30)
    parser.add_argument("--w-far", type=float, default=0.30)
    parser.add_argument("--w-edge", type=float, default=0.20)
    parser.add_argument("--w-iqr", type=float, default=0.20)
    parser.add_argument("--score-min", type=float, default=0.45)

    parser.add_argument("--max-start-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-actions-per-start", type=int, default=30)
    parser.add_argument("--turn-small-deg", type=float, default=10.0)
    parser.add_argument("--forward-small-m", type=float, default=0.2)
    parser.add_argument("--forward-medium-m", type=float, default=0.4)
    parser.add_argument("--back-small-m", type=float, default=0.2)

    parser.add_argument("--cam-height-m", type=float, default=1.5)
    parser.add_argument("--raycast-top-pad-m", type=float, default=5.0)
    parser.add_argument("--min-floor-normal-dot", type=float, default=0.8)
    parser.add_argument("--clearance-min-m", type=float, default=0.4)
    parser.add_argument("--allow-floor-proxy", dest="allow_floor_proxy", action="store_true", default=False)
    parser.add_argument("--no-allow-floor-proxy", dest="allow_floor_proxy", action="store_false")
    parser.add_argument("--forward-hit-gate", dest="forward_hit_gate", action="store_true", default=False)
    parser.add_argument("--no-forward-hit-gate", dest="forward_hit_gate", action="store_false")
    parser.add_argument("--forward-hit-max-m", type=float, default=5.0)
    parser.add_argument("--forward-yaw-trials", type=int, default=12)
    parser.add_argument("--adaptive-forward-hit-max", dest="adaptive_forward_hit_max", action="store_true", default=True)
    parser.add_argument("--no-adaptive-forward-hit-max", dest="adaptive_forward_hit_max", action="store_false")
    parser.add_argument("--forward-hit-max-scale", type=float, default=0.05)
    parser.add_argument("--forward-hit-max-cap-m", type=float, default=30.0)

    args = parser.parse_args()

    args.step0_root = args.step0_root.resolve()
    if args.step1_root is None:
        args.step1_root = args.step0_root.parent / STEP1_DIRNAME
    args.step1_root = args.step1_root.resolve()
    args.step1_root.mkdir(parents=True, exist_ok=True)

    if args.worker_scene_init is not None:
        run_worker_entry(args)
        return

    scene_init_paths = discover_scene_init_paths(step0_root=args.step0_root, scene_id_filter=args.scene_id)
    if not scene_init_paths:
        print(f"No step0 scene_init.json found under: {args.step0_root}")
        return

    summary_path = args.step1_root / SUMMARY_PATH_NAME
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
    inventory_total = len(scene_init_paths)

    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
            f.flush()

        for loop_i, scene_init_path in enumerate(scene_init_paths, start=1):
            step0_payload = load_json(scene_init_path) or {}
            scene_id = scene_id_from_scene_init_path(scene_init_path=scene_init_path, payload=step0_payload)

            if scene_id in done_scene_ids:
                print(f"[{loop_i}/{inventory_total}] SKIP {scene_id} (already in summary)")
                continue
            if args.max_new > 0 and processed_new >= args.max_new:
                print(f"Reached --max-new={args.max_new}, stop.")
                break

            if args.subprocess_isolation:
                row = run_scene_with_subprocess(
                    scene_init_path=scene_init_path,
                    idx=loop_i,
                    total=inventory_total,
                    step1_root=args.step1_root,
                    env_meta=env_meta,
                    args=args,
                )
            else:
                row = run_scene_inline(
                    scene_init_path=scene_init_path,
                    idx=loop_i,
                    total=inventory_total,
                    step1_root=args.step1_root,
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
                "step0_scene_init": row["step0_scene_init"],
                "status": row["status"],
                "run_state": row["run_state"],
                "elapsed_sec": row["elapsed_sec"],
                "step1_ok": row["step1_ok"],
                "start_mode": row["start_mode"],
                "navmesh_status": row["navmesh_status"],
                "candidates_total": row["candidates_total"],
                "accepted_candidates": row["accepted_candidates"],
                "best_score": row["best_score"],
                "fail_reason": row["fail_reason"],
                "step1_report_json": row["step1_report_json"],
                "log_path": row["log_path"],
                "worker_exit_code": row["worker_exit_code"],
            }
            writer.writerow(row_out)
            f.flush()

            done_scene_ids.add(scene_id)
            processed_new += 1
            print(
                f"[{row['idx']}/{row['total']}] {scene_id} | {row['status']} | run_state={row['run_state']} "
                f"| mode={row['start_mode']} | candidates={row['accepted_candidates']}/{row['candidates_total']} "
                f"| best={row['best_score']} | t={row['elapsed_sec']}s | fail={row['fail_reason']}"
            )

    print(
        f"Step1 batch done: inventory_total={inventory_total}, newly_processed={processed_new}, "
        f"resume={'on' if args.resume else 'off'}, subprocess_isolation={'on' if args.subprocess_isolation else 'off'}"
    )


if __name__ == "__main__":
    main()
