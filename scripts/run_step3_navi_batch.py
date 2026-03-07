#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import habitat_sim
import numpy as np
from PIL import Image

from run_step3_batch import (
    DEFAULT_STEP2_ROOT,
    LOG_DIRNAME,
    ROOT,
    SUMMARY_FIELDS,
    SUMMARY_PATH_NAME,
    angle_diff_rad,
    append_jsonl,
    build_sim,
    clamp,
    collect_env_meta,
    dependency_gate,
    depth_proxy_stats,
    discover_step2_reports,
    encode_gif_from_frames,
    encode_video_from_frames,
    extract_start_pose,
    grid_cell,
    load_json,
    make_step3_debug_strip,
    make_visited_map,
    normalize_summary_schema,
    observe_pose,
    parse_step3_report_summary,
    resolve_sensor_config,
    save_json,
    scene_id_from_step2_report,
    synthesize_step3_report_for_crash,
    try_load_navmesh_from_step2,
    validate_scene_source_assets,
    wrap_angle_rad,
)

DEFAULT_OUTPUT_ROOT = ROOT / "output"
STEP3_DIRNAME = "step3_navi"
UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


def yaw_to_forward(yaw_rad: float) -> np.ndarray:
    return np.array([math.sin(float(yaw_rad)), 0.0, -math.cos(float(yaw_rad))], dtype=np.float64)


def forward_to_yaw(vec: np.ndarray) -> float:
    return wrap_angle_rad(math.atan2(float(vec[0]), float(-vec[2])))


def rotate_y(vec: np.ndarray, deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c = math.cos(rad)
    s = math.sin(rad)
    x = float(vec[0])
    z = float(vec[2])
    out = np.array([c * x - s * z, 0.0, s * x + c * z], dtype=np.float64)
    out_h = normalize_horizontal(out)
    if out_h is None:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return out_h


def normalize_horizontal(vec: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float64).copy()
    if arr.shape[0] < 3:
        return None
    arr[1] = 0.0
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def vec3_to_np(v) -> np.ndarray:
    return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)


def safe_snap_nav_point(pathfinder, point: np.ndarray) -> Optional[np.ndarray]:
    try:
        snapped = pathfinder.snap_point(np.asarray(point, dtype=np.float32))
    except Exception:
        return None
    arr = vec3_to_np(snapped)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def obstacle_distance(pathfinder, point: np.ndarray, radius: float) -> Optional[float]:
    try:
        dist = float(pathfinder.distance_to_closest_obstacle(np.asarray(point, dtype=np.float32), float(radius)))
    except Exception:
        return None
    if not math.isfinite(dist):
        return None
    return dist


def obstacle_hit(pathfinder, point: np.ndarray, radius: float) -> Optional[Dict]:
    try:
        hit = pathfinder.closest_obstacle_surface_point(np.asarray(point, dtype=np.float32), float(radius))
    except Exception:
        return None
    dist = float(getattr(hit, "hit_dist", float("nan")))
    hit_pos = vec3_to_np(getattr(hit, "hit_pos", [float("nan")] * 3))
    hit_normal = vec3_to_np(getattr(hit, "hit_normal", [float("nan")] * 3))
    normal_h = normalize_horizontal(hit_normal)
    if (not math.isfinite(dist)) or (not np.all(np.isfinite(hit_pos))) or normal_h is None:
        return None
    if dist >= float(radius) - 1e-3:
        return None
    return {
        "dist": dist,
        "hit_pos": hit_pos,
        "hit_normal": hit_normal,
        "normal_h": normal_h,
    }


def shortest_path_points(pathfinder, start: np.ndarray, goal: np.ndarray) -> Optional[Dict]:
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.asarray(start, dtype=np.float32)
    sp.requested_end = np.asarray(goal, dtype=np.float32)
    try:
        ok = bool(pathfinder.find_path(sp))
    except Exception:
        return None
    if not ok:
        return None
    points = [vec3_to_np(p) for p in list(sp.points)]
    if len(points) < 2:
        return None
    return {
        "points": points,
        "geodesic_distance": float(sp.geodesic_distance),
    }


def densify_nav_path(points: List[np.ndarray], max_step_m: float) -> List[np.ndarray]:
    if len(points) <= 1:
        return list(points)
    out = [np.asarray(points[0], dtype=np.float64)]
    step = max(1e-3, float(max_step_m))
    for i in range(1, len(points)):
        a = np.asarray(points[i - 1], dtype=np.float64)
        b = np.asarray(points[i], dtype=np.float64)
        seg = b - a
        dist = float(np.linalg.norm(seg))
        if dist <= step:
            out.append(b)
            continue
        direction = seg / dist
        n = int(math.ceil(dist / step))
        for j in range(1, n + 1):
            t = min(1.0, float(j) / float(n))
            out.append(a + direction * (dist * t))
    return out


def ensure_same_island(pathfinder, point: np.ndarray, island_id: int) -> bool:
    try:
        return int(pathfinder.get_island(np.asarray(point, dtype=np.float32))) == int(island_id)
    except Exception:
        return False


def estimate_path_novelty(points: List[np.ndarray], visited: set, grid_res_m: float) -> float:
    if len(points) <= 1:
        return 0.0
    cells = []
    for p in points[1:]:
        cells.append(grid_cell(pos=p, grid_res_m=float(grid_res_m)))
    if not cells:
        return 0.0
    new_count = sum(1 for cell in cells if cell not in visited)
    return float(new_count / len(cells))


def recent_overlap_penalty(points: List[np.ndarray], recent_points: Deque[np.ndarray], radius_m: float) -> float:
    if not recent_points or len(points) <= 1:
        return 0.0
    radius = float(max(1e-3, radius_m))
    penalties = []
    for p in points[1:]:
        best = min(float(np.linalg.norm(p - q)) for q in recent_points)
        penalties.append(1.0 if best < radius else 0.0)
    if not penalties:
        return 0.0
    return float(sum(penalties) / len(penalties))


def compute_camera_height(step2_payload: Dict, start_nav_y: float, args) -> float:
    base = step2_payload.get("cam_y_minus_floor_height")
    if base is None:
        base = step2_payload.get("cam_height_above_floor")
    if base is None:
        pose_before = step2_payload.get("floor_pose_apply_readback_before")
        if isinstance(pose_before, dict):
            pos = pose_before.get("position")
            if isinstance(pos, list) and len(pos) >= 3:
                base = float(pos[1]) - float(start_nav_y)
    if base is None:
        base = 1.5
    camera_height = float(base) + float(args.camera_height_bias_m)
    return clamp(camera_height, float(args.camera_height_min_m), float(args.camera_height_max_m))


def build_camera_pose(nav_pos: np.ndarray, camera_height_m: float) -> np.ndarray:
    cam = np.asarray(nav_pos, dtype=np.float64).copy()
    cam[1] = float(nav_pos[1]) + float(camera_height_m)
    return cam


def choose_wall_tangent(wall_normal_h: np.ndarray, wall_side: str, ref_dir: np.ndarray) -> Optional[np.ndarray]:
    if wall_side == "left":
        tangent = np.cross(UP, wall_normal_h)
    else:
        tangent = np.cross(wall_normal_h, UP)
    tangent_h = normalize_horizontal(tangent)
    if tangent_h is None:
        return None
    if ref_dir is not None and float(np.dot(tangent_h, ref_dir)) < 0.0:
        tangent_h = -tangent_h
    return tangent_h


def ordered_offsets(max_turn_deg: float, step_deg: float, include_reverse: bool) -> List[float]:
    max_turn_deg = float(max(0.0, max_turn_deg))
    step_deg = float(max(1.0, step_deg))
    out = [0.0]
    cur = step_deg
    while cur <= max_turn_deg + 1e-6:
        out.extend([-cur, cur])
        cur += step_deg
    if include_reverse:
        out.extend([-180.0, 180.0])
    seen = set()
    unique = []
    for v in out:
        key = round(float(v), 4)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(v))
    return unique


def build_local_wall_plan(
    pathfinder,
    nav_pos: np.ndarray,
    yaw_rad: float,
    last_motion_dir: Optional[np.ndarray],
    visited: set,
    recent_nav_points: Deque[np.ndarray],
    args,
) -> Optional[Dict]:
    ref_dir = last_motion_dir if last_motion_dir is not None else yaw_to_forward(yaw_rad)
    try:
        island_id = int(pathfinder.get_island(np.asarray(nav_pos, dtype=np.float32)))
    except Exception:
        island_id = -1

    wall = obstacle_hit(pathfinder=pathfinder, point=nav_pos, radius=float(args.wall_search_radius_m))
    candidates = []

    if wall is not None:
        tangent = choose_wall_tangent(
            wall_normal_h=wall["normal_h"],
            wall_side=str(args.wall_side),
            ref_dir=ref_dir,
        )
        correction = wall["normal_h"] * clamp(
            (float(args.wall_target_dist_m) - float(wall["dist"])) * float(args.wall_correction_gain),
            -float(args.wall_correction_max_m),
            float(args.wall_correction_max_m),
        )
        if tangent is not None:
            candidates.append(("wall_tangent", tangent, correction))
    if not candidates:
        candidates.append(("free_scan", ref_dir, np.zeros(3, dtype=np.float64)))

    offsets = ordered_offsets(
        max_turn_deg=float(args.candidate_max_turn_deg),
        step_deg=float(args.candidate_angle_step_deg),
        include_reverse=(wall is None),
    )
    best = None
    score_best = -1e18

    for mode_name, base_dir, correction in candidates:
        for offset_deg in offsets:
            dir_vec = rotate_y(base_dir, offset_deg)
            if dir_vec is None:
                continue
            for probe_dist in args.goal_probe_dists_m:
                raw_goal = np.asarray(nav_pos, dtype=np.float64) + dir_vec * float(probe_dist) + correction
                snapped_goal = safe_snap_nav_point(pathfinder=pathfinder, point=raw_goal)
                if snapped_goal is None:
                    continue
                if bool(args.same_island_only) and island_id >= 0 and not ensure_same_island(pathfinder, snapped_goal, island_id):
                    continue
                path = shortest_path_points(pathfinder=pathfinder, start=nav_pos, goal=snapped_goal)
                if path is None:
                    continue
                geodesic_distance = float(path["geodesic_distance"])
                if geodesic_distance < float(args.local_goal_min_m):
                    continue
                if geodesic_distance > float(args.local_goal_max_m):
                    continue

                dense_path = densify_nav_path(points=path["points"], max_step_m=float(args.nav_step_m))
                if len(dense_path) < 2:
                    continue

                first_move_vec = normalize_horizontal(dense_path[min(1, len(dense_path) - 1)] - dense_path[0])
                if first_move_vec is None:
                    continue

                goal_wall_dist = obstacle_distance(
                    pathfinder=pathfinder,
                    point=snapped_goal,
                    radius=float(args.wall_search_radius_m),
                )
                if goal_wall_dist is None:
                    goal_wall_dist = float(args.wall_search_radius_m)

                novelty = estimate_path_novelty(
                    points=dense_path,
                    visited=visited,
                    grid_res_m=float(args.grid_res_m),
                )
                overlap = recent_overlap_penalty(
                    points=dense_path,
                    recent_points=recent_nav_points,
                    radius_m=float(args.recent_overlap_radius_m),
                )
                align = float(np.dot(first_move_vec, dir_vec))
                turn_penalty = abs(
                    math.degrees(angle_diff_rad(forward_to_yaw(first_move_vec), forward_to_yaw(ref_dir)))
                ) / 180.0
                wall_score = 1.0 - min(
                    1.0,
                    abs(float(goal_wall_dist) - float(args.wall_target_dist_m)) / max(float(args.wall_target_dist_m), 0.35),
                )
                if float(goal_wall_dist) >= float(args.wall_open_penalty_m):
                    wall_score -= 0.35
                geo_score = min(1.0, geodesic_distance / max(float(args.local_goal_max_m), 1e-3))

                score = (
                    1.8 * align
                    + 1.4 * wall_score
                    + 1.2 * novelty
                    + 0.25 * geo_score
                    - 0.9 * overlap
                    - 0.45 * turn_penalty
                )
                if score > score_best:
                    score_best = score
                    best = {
                        "mode": mode_name,
                        "goal_nav": snapped_goal,
                        "path_points": dense_path,
                        "score": float(score),
                        "geodesic_distance": geodesic_distance,
                        "wall_dist_now": None if wall is None else float(wall["dist"]),
                        "wall_dist_goal": float(goal_wall_dist),
                        "offset_deg": float(offset_deg),
                        "probe_dist_m": float(probe_dist),
                    }

    if best is not None:
        best["candidate_count"] = len(offsets) * len(args.goal_probe_dists_m)
        best["wall_available"] = bool(wall is not None)
    return best


def run_step3_scene_worker(step2_report_path: Path, step3_root: Path, args, env_meta: Dict) -> Dict:
    step2_payload = load_json(step2_report_path)
    scene_id = scene_id_from_step2_report(step2_report_path, step2_payload)
    scene_dir = step3_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    report_path = scene_dir / "step3_report.json"
    traj_meta_path = scene_dir / "traj_meta.json"
    poses_path = scene_dir / "poses.jsonl"
    frames_dir = scene_dir / "frames"
    video_path = scene_dir / "step3_frames.mp4"
    video_gif_path = scene_dir / "step3_frames.gif"

    if poses_path.exists():
        poses_path.unlink()
    if bool(args.export_frames):
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("frame_*.png"):
            try:
                old.unlink()
            except Exception:
                pass
    if bool(args.make_video) and video_path.exists():
        try:
            video_path.unlink()
        except Exception:
            pass
    if bool(args.video_fallback_gif) and video_gif_path.exists():
        try:
            video_gif_path.unlink()
        except Exception:
            pass

    if not isinstance(step2_payload, dict):
        payload = {
            "scene_id": scene_id,
            "pipeline_stage": 3,
            "step_name": "step3_navi_walltrace",
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "fail_reason": "STEP2_REPORT_MISSING_OR_INVALID",
            "step2_report": str(step2_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        save_json(traj_meta_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": "",
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "frames": 0,
            "path_length_m": 0.0,
            "visited_unique_cells": 0,
            "coverage_area_m2": 0.0,
            "continuity_valid_ratio": 0.0,
            "recovery_ratio": 0.0,
            "room_check_triggers_total": 0,
            "escape_attempts_total": 0,
            "fail_reason": "STEP2_REPORT_MISSING_OR_INVALID",
            "traj_meta_json": str(traj_meta_path),
        }

    dep_ok, dep_reason = dependency_gate(step2_payload)
    scene_path = str(step2_payload.get("scene_path") or "")
    if not dep_ok:
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 3,
            "step_name": "step3_navi_walltrace",
            "status": "SKIP",
            "run_state": "SKIP",
            "step3_ok": False,
            "fail_reason": dep_reason,
            "step2_report": str(step2_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        save_json(traj_meta_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "SKIP",
            "run_state": "SKIP",
            "step3_ok": False,
            "frames": 0,
            "path_length_m": 0.0,
            "visited_unique_cells": 0,
            "coverage_area_m2": 0.0,
            "continuity_valid_ratio": 0.0,
            "recovery_ratio": 0.0,
            "room_check_triggers_total": 0,
            "escape_attempts_total": 0,
            "fail_reason": dep_reason,
            "traj_meta_json": str(traj_meta_path),
        }

    scene_source_raw = step2_payload.get("scene_source")
    if not isinstance(scene_source_raw, str) or not scene_source_raw.strip():
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 3,
            "step_name": "step3_navi_walltrace",
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "fail_reason": "step2_scene_source_missing",
            "step2_report": str(step2_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        save_json(traj_meta_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "frames": 0,
            "path_length_m": 0.0,
            "visited_unique_cells": 0,
            "coverage_area_m2": 0.0,
            "continuity_valid_ratio": 0.0,
            "recovery_ratio": 0.0,
            "room_check_triggers_total": 0,
            "escape_attempts_total": 0,
            "fail_reason": "step2_scene_source_missing",
            "traj_meta_json": str(traj_meta_path),
        }
    scene_source = Path(scene_source_raw)
    source_ok, source_reason, source_detail = validate_scene_source_assets(scene_source=scene_source)
    if not source_ok:
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 3,
            "step_name": "step3_navi_walltrace",
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "fail_reason": "step3_scene_source_invalid",
            "fail_reason_detail": source_reason,
            "scene_source_validation": source_detail,
            "step2_report": str(step2_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        save_json(traj_meta_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "frames": 0,
            "path_length_m": 0.0,
            "visited_unique_cells": 0,
            "coverage_area_m2": 0.0,
            "continuity_valid_ratio": 0.0,
            "recovery_ratio": 0.0,
            "room_check_triggers_total": 0,
            "escape_attempts_total": 0,
            "fail_reason": "step3_scene_source_invalid",
            "traj_meta_json": str(traj_meta_path),
        }

    start_pos, start_yaw, start_pitch, start_err = extract_start_pose(step2_payload=step2_payload, args=args)
    if start_err is not None or start_pos is None or start_yaw is None or start_pitch is None:
        payload = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "pipeline_stage": 3,
            "step_name": "step3_navi_walltrace",
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "fail_reason": "step1_start_missing",
            "fail_reason_detail": start_err,
            "step2_report": str(step2_report_path),
            "environment": env_meta,
        }
        save_json(report_path, payload)
        save_json(traj_meta_path, payload)
        return {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "status": "FAIL",
            "run_state": "FAIL",
            "step3_ok": False,
            "frames": 0,
            "path_length_m": 0.0,
            "visited_unique_cells": 0,
            "coverage_area_m2": 0.0,
            "continuity_valid_ratio": 0.0,
            "recovery_ratio": 0.0,
            "room_check_triggers_total": 0,
            "escape_attempts_total": 0,
            "fail_reason": "step1_start_missing",
            "traj_meta_json": str(traj_meta_path),
        }

    sensor_cfg = resolve_sensor_config(step2_payload=step2_payload, args=args)
    report = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 3,
        "step_name": "step3_navi_walltrace",
        "step2_report": str(step2_report_path),
        "scene_source": str(scene_source),
        "sensor_config": sensor_cfg,
        "environment": env_meta,
        "params": {
            "target_frames": int(args.target_frames),
            "path_target_m": float(args.path_target_m),
            "path_max_m": float(args.path_max_m),
            "path_min_done_m": float(args.path_min_done_m),
            "wall_side": str(args.wall_side),
            "wall_target_dist_m": float(args.wall_target_dist_m),
            "nav_step_m": float(args.nav_step_m),
            "camera_height_bias_m": float(args.camera_height_bias_m),
        },
    }

    sim = None
    t0 = time.time()
    timings = {}
    status = "FAIL"
    run_state = "FAIL"
    step3_ok = False
    fail_reason = "unknown"
    fail_reason_detail = None

    path_length_m = 0.0
    continuity_ok_frames = 0
    recovery_frames = 0
    frames = 0
    visited = set()
    visited_unique_cells = 0
    coverage_area_m2 = 0.0
    room_check_triggers_total = 0
    escape_attempts_total = 0

    blocked_streak = 0
    poor_view_streak = 0
    no_plan_streak = 0
    origin_match_streak = 0
    replans_total = 0
    blocked_replans = 0
    poor_view_replans = 0
    completed_replans = 0
    wall_missing_frames = 0
    successful_motion_frames = 0
    max_blocked_streak = 0
    max_poor_view_streak = 0
    fail_reason_hist = Counter()
    warnings = []
    debug_images = []
    recent_nav_points: Deque[np.ndarray] = deque(maxlen=int(args.recent_nav_window))
    plateau_window = deque(maxlen=int(args.coverage_plateau_window_frames))

    active_plan = None
    active_path_points: List[np.ndarray] = []
    active_path_idx = 0
    last_motion_dir = None
    last_plan_reason = "startup"
    plateau_reason = None
    terminal_gate = None
    motion_state = "PLAN"

    try:
        ts = time.time()
        scene_seed = int((int(args.seed) + (abs(hash(scene_id)) % 2147483647)) % 2147483647)
        sim = build_sim(scene_source=scene_source, sensor_cfg=sensor_cfg, seed=scene_seed, enable_physics=(not args.disable_physics))
        agent = sim.initialize_agent(0)
        timings["t_load_scene"] = float(time.time() - ts)

        ts = time.time()
        navmesh_probe = try_load_navmesh_from_step2(sim=sim, step2_payload=step2_payload)
        timings["t_load_navmesh"] = float(time.time() - ts)
        report["navmesh_probe"] = navmesh_probe
        report["use_navmesh_move"] = bool(navmesh_probe.get("loaded", False))
        if not bool(navmesh_probe.get("loaded", False)):
            raise RuntimeError(f"navmesh_required:{navmesh_probe.get('error')}")

        start_nav = safe_snap_nav_point(pathfinder=sim.pathfinder, point=np.asarray(start_pos, dtype=np.float64))
        if start_nav is None:
            raise RuntimeError("start_nav_snap_failed")

        camera_height_m = compute_camera_height(step2_payload=step2_payload, start_nav_y=float(start_nav[1]), args=args)
        nav_pos = start_nav.copy()
        yaw = float(start_yaw)
        pitch = float(start_pitch)
        origin_nav = nav_pos.copy()
        camera_pos = build_camera_pose(nav_pos=nav_pos, camera_height_m=camera_height_m)

        report["start_pose"] = {
            "camera_position": camera_pos.tolist(),
            "nav_position": nav_pos.tolist(),
            "yaw_rad": float(yaw),
            "pitch_rad": float(pitch),
            "camera_height_m": float(camera_height_m),
        }

        done_reason = None
        max_frames_guard = int(max(args.target_frames * 2, args.target_frames + 40))

        for frame_idx in range(max_frames_guard):
            frame_t0 = time.time()
            camera_pos = build_camera_pose(nav_pos=nav_pos, camera_height_m=camera_height_m)

            ts_render = time.time()
            obs = observe_pose(
                sim=sim,
                agent=agent,
                position=camera_pos.astype(np.float32),
                yaw_rad=yaw,
                pitch_rad=pitch,
            )
            rgb = np.asarray(obs["rgb"], dtype=np.uint8)
            depth = np.asarray(obs["depth"], dtype=np.float32)
            timings_render = float(time.time() - ts_render)

            if bool(args.export_frames):
                Image.fromarray(rgb).save(frames_dir / f"frame_{frame_idx:04d}.png")
            if frame_idx % 10 == 0:
                debug_images.append(rgb.copy())

            stats = depth_proxy_stats(depth=depth, args=args)
            valid_ratio = float(stats["valid_ratio"])
            black_ratio = float(np.mean(np.all(rgb <= int(args.black_pixel_threshold), axis=2)))

            poor_view = bool(
                valid_ratio < float(args.poor_view_valid_ratio_min)
                or black_ratio > float(args.poor_view_black_ratio_max)
            )
            if poor_view:
                poor_view_streak += 1
            else:
                poor_view_streak = 0
            max_poor_view_streak = max(max_poor_view_streak, poor_view_streak)

            cur_cell = grid_cell(pos=nav_pos, grid_res_m=float(args.grid_res_m))
            visited_before = len(visited)
            visited.add(cur_cell)
            visited_unique_cells = len(visited)
            coverage_area_m2 = float(visited_unique_cells) * float(args.grid_res_m) * float(args.grid_res_m)
            new_cell = 1 if visited_unique_cells > visited_before else 0
            plateau_window.append({"visited": visited_unique_cells, "path": path_length_m})
            recent_nav_points.append(nav_pos.copy())

            wall_now = obstacle_hit(pathfinder=sim.pathfinder, point=nav_pos, radius=float(args.wall_search_radius_m))
            if wall_now is None:
                wall_missing_frames += 1

            replan_reason = None
            if poor_view_streak >= int(args.poor_view_replan_streak):
                replan_reason = "poor_view"
            elif not active_path_points or active_path_idx >= len(active_path_points):
                replan_reason = "plan_exhausted"
            elif blocked_streak >= int(args.blocked_streak_replan):
                replan_reason = "blocked"

            if replan_reason is not None:
                ts_policy = time.time()
                active_plan = build_local_wall_plan(
                    pathfinder=sim.pathfinder,
                    nav_pos=nav_pos,
                    yaw_rad=yaw,
                    last_motion_dir=last_motion_dir,
                    visited=visited,
                    recent_nav_points=recent_nav_points,
                    args=args,
                )
                timings_policy = float(time.time() - ts_policy)
                if active_plan is None:
                    no_plan_streak += 1
                    active_path_points = []
                    active_path_idx = 0
                    if no_plan_streak >= int(args.no_plan_fail_streak):
                        fail_reason = "navi_no_plan"
                        fail_reason_detail = f"replan_reason={replan_reason}, no_plan_streak={no_plan_streak}"
                        terminal_gate = "planner"
                        if path_length_m >= float(args.path_min_done_m):
                            done_reason = "TIMEOUT"
                        else:
                            done_reason = "FAIL"
                        break
                else:
                    no_plan_streak = 0
                    active_path_points = list(active_plan["path_points"])
                    active_path_idx = 1
                    replans_total += 1
                    escape_attempts_total = replans_total
                    last_plan_reason = replan_reason
                    motion_state = active_plan["mode"]
                    recovery_frames += 1
                    if replan_reason == "blocked":
                        blocked_replans += 1
                    elif replan_reason == "poor_view":
                        poor_view_replans += 1
                    elif replan_reason == "plan_exhausted":
                        completed_replans += 1
            else:
                timings_policy = 0.0

            while active_path_idx < len(active_path_points):
                if float(np.linalg.norm(active_path_points[active_path_idx] - nav_pos)) <= float(args.waypoint_reach_m):
                    active_path_idx += 1
                    continue
                break

            if active_path_idx >= len(active_path_points):
                active_path_points = []
                motion_state = "PLAN"
                nav_next = nav_pos.copy()
                step_dist = 0.0
                desired_yaw = yaw
                recovery_used = "none"
            else:
                target_nav = active_path_points[active_path_idx]
                move_vec = target_nav - nav_pos
                move_dir = normalize_horizontal(move_vec)
                if move_dir is None:
                    nav_next = nav_pos.copy()
                    step_dist = 0.0
                    desired_yaw = yaw
                    recovery_used = "none"
                else:
                    desired_yaw = forward_to_yaw(move_dir)
                    step_target = nav_pos + move_dir * min(float(args.nav_step_m), float(np.linalg.norm(move_vec)))
                    try:
                        stepped = sim.pathfinder.try_step(
                            np.asarray(nav_pos, dtype=np.float32),
                            np.asarray(step_target, dtype=np.float32),
                        )
                        nav_next = vec3_to_np(stepped)
                    except Exception:
                        nav_next = nav_pos.copy()
                    step_dist = float(np.linalg.norm(nav_next - nav_pos))
                    recovery_used = "replan" if replan_reason is not None else "none"

            if step_dist >= float(args.blocked_delta_pos_m):
                blocked_streak = 0
                successful_motion_frames += 1
                last_motion_dir = normalize_horizontal(nav_next - nav_pos)
            else:
                blocked_streak += 1
            max_blocked_streak = max(max_blocked_streak, blocked_streak)

            yaw_delta = angle_diff_rad(desired_yaw, yaw)
            yaw += clamp(yaw_delta, -math.radians(float(args.max_yaw_step_deg)), math.radians(float(args.max_yaw_step_deg)))
            yaw = wrap_angle_rad(yaw)

            path_length_m += step_dist
            nav_pos = nav_next
            frames = frame_idx + 1

            if step_dist >= float(args.blocked_delta_pos_m):
                continuity_ok_frames += 1

            coverage_plateau = False
            coverage_gain_last_window = 0
            if len(plateau_window) >= int(args.coverage_plateau_window_frames):
                oldest = plateau_window[0]
                coverage_gain_last_window = int(visited_unique_cells - int(oldest["visited"]))
                coverage_plateau = bool(
                    path_length_m >= float(args.path_target_m)
                    and coverage_gain_last_window <= int(args.coverage_plateau_new_cells_max)
                )

            origin_dist = float(np.linalg.norm(nav_pos - origin_nav))
            origin_yaw_diff = abs(math.degrees(angle_diff_rad(yaw, float(start_yaw))))
            if (
                path_length_m >= float(args.path_min_done_m)
                and origin_dist < float(args.origin_finish_pos_m)
                and origin_yaw_diff < float(args.origin_finish_yaw_deg)
            ):
                origin_match_streak += 1
            else:
                origin_match_streak = 0

            if poor_view_streak >= int(args.poor_view_fail_streak):
                fail_reason = "outside_scene_or_blackframe"
                fail_reason_detail = (
                    f"poor_view_streak={poor_view_streak}, valid_ratio={valid_ratio:.3f}, black_ratio={black_ratio:.3f}"
                )
                terminal_gate = "poor_view"
                if path_length_m >= float(args.path_min_done_m):
                    done_reason = "TIMEOUT"
                else:
                    done_reason = "FAIL"
                frame_elapsed = float(time.time() - frame_t0)
                append_jsonl(
                    poses_path,
                    {
                        "frame_idx": frame_idx,
                        "fsm_state": motion_state,
                        "replan_reason": replan_reason,
                        "continuity_status": "FAIL",
                        "recovery_used": recovery_used,
                        "path_length_m": path_length_m,
                        "coverage_area_m2": coverage_area_m2,
                        "poor_view": poor_view,
                        "black_ratio": black_ratio,
                        "depth_valid_ratio": valid_ratio,
                        "pose": {
                            "nav_position": nav_pos.tolist(),
                            "camera_position": build_camera_pose(nav_pos=nav_pos, camera_height_m=camera_height_m).tolist(),
                            "yaw_rad": float(yaw),
                            "pitch_rad": float(pitch),
                        },
                        "stage_timing_sec": {
                            "t_step3_render": timings_render,
                            "t_step3_policy": timings_policy,
                            "t_step3_total": frame_elapsed,
                        },
                    },
                )
                break

            frame_elapsed = float(time.time() - frame_t0)
            frame_record = {
                "frame_idx": frame_idx,
                "fsm_state": motion_state,
                "replan_reason": replan_reason,
                "plan_mode": None if active_plan is None else active_plan.get("mode"),
                "plan_score": None if active_plan is None else active_plan.get("score"),
                "plan_goal_nav": None if active_plan is None else active_plan.get("goal_nav").tolist(),
                "plan_geodesic_distance_m": None if active_plan is None else active_plan.get("geodesic_distance"),
                "wall_dist_now_m": None if wall_now is None else float(wall_now["dist"]),
                "blocked_streak": int(blocked_streak),
                "poor_view_streak": int(poor_view_streak),
                "depth_valid_ratio": valid_ratio,
                "black_ratio": black_ratio,
                "continuity_status": "OK" if step_dist >= float(args.blocked_delta_pos_m) else "RECOVERY_OK",
                "recovery_used": recovery_used,
                "motion_source": "navmesh_path",
                "step_dist_m": float(step_dist),
                "path_length_m": float(path_length_m),
                "visited_unique_cells": int(visited_unique_cells),
                "coverage_area_m2": float(coverage_area_m2),
                "new_cell": int(new_cell),
                "coverage_gain_last_window": int(coverage_gain_last_window),
                "coverage_plateau": bool(coverage_plateau),
                "pose": {
                    "nav_position": nav_pos.tolist(),
                    "camera_position": build_camera_pose(nav_pos=nav_pos, camera_height_m=camera_height_m).tolist(),
                    "yaw_rad": float(yaw),
                    "pitch_rad": float(pitch),
                },
                "stage_timing_sec": {
                    "t_step3_render": timings_render,
                    "t_step3_policy": timings_policy,
                    "t_step3_total": frame_elapsed,
                },
            }
            append_jsonl(poses_path, frame_record)

            if origin_match_streak >= int(args.origin_finish_streak):
                done_reason = "DONE"
                plateau_reason = "origin_finish"
                terminal_gate = "origin_finish"
                break
            if path_length_m >= float(args.path_max_m):
                done_reason = "DONE"
                plateau_reason = "path_max_reached"
                terminal_gate = "path_max"
                break
            if coverage_plateau:
                done_reason = "DONE"
                plateau_reason = "coverage_plateau"
                terminal_gate = "coverage_plateau"
                break
            if frames >= int(args.target_frames):
                if path_length_m >= float(args.path_target_m):
                    done_reason = "DONE"
                    plateau_reason = "frame_budget_path_target"
                    terminal_gate = "frame_budget_done"
                elif path_length_m >= float(args.path_min_done_m):
                    done_reason = "TIMEOUT"
                    fail_reason = "target_frames_reached"
                    fail_reason_detail = f"path_length_m={path_length_m:.3f}<path_target_m={float(args.path_target_m):.3f}"
                    terminal_gate = "frame_budget_timeout"
                else:
                    done_reason = "FAIL"
                    fail_reason = "stuck_no_progress"
                    fail_reason_detail = (
                        f"path_length_m={path_length_m:.3f}<path_min_done_m={float(args.path_min_done_m):.3f}"
                    )
                    terminal_gate = "frame_budget_fail"
                break

        if done_reason == "DONE":
            status = "OK"
            run_state = "DONE"
            step3_ok = True
            fail_reason = None
        elif done_reason == "TIMEOUT":
            status = "FAIL"
            run_state = "TIMEOUT"
            step3_ok = False
            if fail_reason is None:
                fail_reason = "target_frames_reached"
        else:
            status = "FAIL"
            run_state = "FAIL"
            step3_ok = False
            if fail_reason is None:
                fail_reason = "stuck_no_progress"

    except Exception as exc:
        status = "FAIL"
        run_state = "FAIL"
        step3_ok = False
        fail_reason = "STEP3_EXCEPTION"
        fail_reason_detail = str(exc)
        navmesh_probe = {
            "loaded": False,
            "source": "exception",
            "cache_path": None,
            "error": str(exc),
        }
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass

    continuity_valid_ratio = float(continuity_ok_frames / max(frames, 1))
    recovery_ratio = float(recovery_frames / max(frames, 1))
    coverage_area_m2 = float(len(visited)) * float(args.grid_res_m) * float(args.grid_res_m)

    if not step3_ok and fail_reason:
        fail_reason_hist[fail_reason] += 1

    traj_meta = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 3,
        "step_name": "step3_navi_walltrace",
        "step2_report": str(step2_report_path),
        "status": status,
        "run_state": run_state,
        "step3_ok": bool(step3_ok),
        "navmesh_probe": navmesh_probe,
        "frames": int(frames),
        "path_length_m": float(path_length_m),
        "visited_unique_cells": int(len(visited)),
        "coverage_area_m2": float(coverage_area_m2),
        "continuity_valid_ratio": float(continuity_valid_ratio),
        "recovery_ratio": float(recovery_ratio),
        "replans_total": int(replans_total),
        "blocked_replans": int(blocked_replans),
        "poor_view_replans": int(poor_view_replans),
        "completed_replans": int(completed_replans),
        "wall_missing_frames": int(wall_missing_frames),
        "successful_motion_frames": int(successful_motion_frames),
        "max_blocked_streak": int(max_blocked_streak),
        "max_poor_view_streak": int(max_poor_view_streak),
        "room_check_triggers_total": 0,
        "escape_attempts_total": int(replans_total),
        "last_plan_reason": last_plan_reason,
        "plateau_reason": plateau_reason,
        "fail_reason_histogram": dict(fail_reason_hist),
        "fail_reason": fail_reason,
        "fail_reason_detail": fail_reason_detail,
        "terminal_record": {
            "last_fsm_state": motion_state,
            "gate_name": terminal_gate,
            "fail_reason": fail_reason,
            "counters_snapshot": {
                "replans_total": int(replans_total),
                "blocked_replans": int(blocked_replans),
                "poor_view_replans": int(poor_view_replans),
                "successful_motion_frames": int(successful_motion_frames),
                "max_blocked_streak": int(max_blocked_streak),
                "max_poor_view_streak": int(max_poor_view_streak),
            },
        },
        "artifacts": {
            "poses_jsonl": str(poses_path),
            "traj_meta_json": str(traj_meta_path),
            "step3_debug_strip": str(scene_dir / "step3_debug_strip.png"),
            "step3_fail_snapshot": str(scene_dir / "step3_fail_snapshot.png"),
            "visited_map_debug": str(scene_dir / "visited_map_debug.png"),
            "frames_dir": str(frames_dir) if bool(args.export_frames) else None,
            "step3_video_mp4": str(video_path) if bool(args.make_video) else None,
            "step3_video_gif": str(video_gif_path) if bool(args.video_fallback_gif) else None,
        },
        "warnings": warnings,
        "environment": env_meta,
    }

    make_step3_debug_strip(debug_images, scene_dir / "step3_debug_strip.png")
    make_visited_map(visited, scene_dir / "visited_map_debug.png")

    if (not step3_ok) and debug_images:
        Image.fromarray(debug_images[-1]).save(scene_dir / "step3_fail_snapshot.png")

    ok_video = True
    video_msg = ""
    if bool(args.make_video):
        ok_video, video_msg = encode_video_from_frames(
            frames_dir=frames_dir,
            out_path=video_path,
            fps=int(args.video_fps),
        )
        if not ok_video:
            warnings.append(video_msg)
    if bool(args.video_fallback_gif):
        ok_gif, gif_msg = encode_gif_from_frames(
            frames_dir=frames_dir,
            out_path=video_gif_path,
            fps=int(args.video_fps),
        )
        if not ok_gif:
            warnings.append(gif_msg)
        elif bool(args.make_video) and not ok_video:
            warnings.append(f"{video_msg}; gif_ok")

    timings["t_total"] = float(time.time() - t0)
    traj_meta["stage_timing_sec"] = timings
    save_json(traj_meta_path, traj_meta)

    report.update(
        {
            "status": status,
            "run_state": run_state,
            "step3_ok": bool(step3_ok),
            "frames": int(frames),
            "path_length_m": float(path_length_m),
            "visited_unique_cells": int(len(visited)),
            "coverage_area_m2": float(coverage_area_m2),
            "continuity_valid_ratio": float(continuity_valid_ratio),
            "recovery_ratio": float(recovery_ratio),
            "room_check_triggers_total": 0,
            "escape_attempts_total": int(replans_total),
            "fail_reason": fail_reason,
            "fail_reason_detail": fail_reason_detail,
            "traj_meta_json": str(traj_meta_path),
            "poses_jsonl": str(poses_path),
            "frames_dir": str(frames_dir) if bool(args.export_frames) else None,
            "step3_video_mp4": str(video_path) if bool(args.make_video) else None,
            "step3_video_gif": str(video_gif_path) if bool(args.video_fallback_gif) else None,
            "warnings": warnings,
            "stage_timing_sec": timings,
        }
    )
    save_json(report_path, report)

    return {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "status": status,
        "run_state": run_state,
        "step3_ok": bool(step3_ok),
        "frames": int(frames),
        "path_length_m": float(path_length_m),
        "visited_unique_cells": int(len(visited)),
        "coverage_area_m2": float(coverage_area_m2),
        "continuity_valid_ratio": float(continuity_valid_ratio),
        "recovery_ratio": float(recovery_ratio),
        "room_check_triggers_total": 0,
        "escape_attempts_total": int(replans_total),
        "fail_reason": fail_reason,
        "traj_meta_json": str(traj_meta_path),
        "step3_report_json": str(report_path),
    }


def build_worker_cmd(script_path: Path, step2_report_path: Path, args) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker-step2-report",
        str(step2_report_path),
        "--step2-root",
        str(args.step2_root),
        "--step3-root",
        str(args.step3_root),
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
        "--target-frames",
        str(args.target_frames),
        "--path-target-m",
        str(args.path_target_m),
        "--path-max-m",
        str(args.path_max_m),
        "--path-min-done-m",
        str(args.path_min_done_m),
        "--grid-res-m",
        str(args.grid_res_m),
        "--wall-side",
        str(args.wall_side),
        "--wall-target-dist-m",
        str(args.wall_target_dist_m),
        "--wall-search-radius-m",
        str(args.wall_search_radius_m),
        "--wall-correction-gain",
        str(args.wall_correction_gain),
        "--wall-correction-max-m",
        str(args.wall_correction_max_m),
        "--wall-open-penalty-m",
        str(args.wall_open_penalty_m),
        "--candidate-angle-step-deg",
        str(args.candidate_angle_step_deg),
        "--candidate-max-turn-deg",
        str(args.candidate_max_turn_deg),
        "--local-goal-min-m",
        str(args.local_goal_min_m),
        "--local-goal-max-m",
        str(args.local_goal_max_m),
        "--nav-step-m",
        str(args.nav_step_m),
        "--waypoint-reach-m",
        str(args.waypoint_reach_m),
        "--blocked-delta-pos-m",
        str(args.blocked_delta_pos_m),
        "--blocked-streak-replan",
        str(args.blocked_streak_replan),
        "--no-plan-fail-streak",
        str(args.no_plan_fail_streak),
        "--poor-view-valid-ratio-min",
        str(args.poor_view_valid_ratio_min),
        "--poor-view-black-ratio-max",
        str(args.poor_view_black_ratio_max),
        "--poor-view-replan-streak",
        str(args.poor_view_replan_streak),
        "--poor-view-fail-streak",
        str(args.poor_view_fail_streak),
        "--recent-nav-window",
        str(args.recent_nav_window),
        "--recent-overlap-radius-m",
        str(args.recent_overlap_radius_m),
        "--camera-height-bias-m",
        str(args.camera_height_bias_m),
        "--camera-height-min-m",
        str(args.camera_height_min_m),
        "--camera-height-max-m",
        str(args.camera_height_max_m),
        "--step3-pitch-deg",
        str(args.step3_pitch_deg),
        "--origin-finish-pos-m",
        str(args.origin_finish_pos_m),
        "--origin-finish-yaw-deg",
        str(args.origin_finish_yaw_deg),
        "--origin-finish-streak",
        str(args.origin_finish_streak),
        "--coverage-plateau-window-frames",
        str(args.coverage_plateau_window_frames),
        "--coverage-plateau-new-cells-max",
        str(args.coverage_plateau_new_cells_max),
        "--max-yaw-step-deg",
        str(args.max_yaw_step_deg),
        "--video-fps",
        str(args.video_fps),
        "--black-pixel-threshold",
        str(args.black_pixel_threshold),
        "--depth-valid-min-m",
        str(args.depth_valid_min_m),
        "--depth-valid-max-m",
        str(args.depth_valid_max_m),
        "--near-wall-depth-m",
        str(args.near_wall_depth_m),
        "--blank-valid-ratio",
        str(args.blank_valid_ratio),
        "--outside-valid-ratio-min",
        str(args.outside_valid_ratio_min),
        "--outside-center-valid-ratio-max",
        str(args.outside_center_valid_ratio_max),
        "--outside-side-valid-ratio-max",
        str(args.outside_side_valid_ratio_max),
        "--no-resume",
    ]
    for v in args.goal_probe_dists_m:
        cmd.extend(["--goal-probe-dists-m", str(v)])
    if args.disable_physics:
        cmd.append("--disable-physics")
    if args.export_frames:
        cmd.append("--export-frames")
    else:
        cmd.append("--no-export-frames")
    if args.make_video:
        cmd.append("--make-video")
    else:
        cmd.append("--no-make-video")
    if args.video_fallback_gif:
        cmd.append("--video-fallback-gif")
    else:
        cmd.append("--no-video-fallback-gif")
    if args.same_island_only:
        cmd.append("--same-island-only")
    else:
        cmd.append("--no-same-island-only")
    if args.enforce_depth_valid_max:
        cmd.append("--enforce-depth-valid-max")
    else:
        cmd.append("--no-enforce-depth-valid-max")
    return cmd


def run_scene_with_subprocess(
    step2_report_path: Path,
    idx: int,
    total: int,
    step3_root: Path,
    env_meta: Dict,
    args,
) -> Dict:
    step2_payload = load_json(step2_report_path) or {}
    scene_id = scene_id_from_step2_report(report_path=step2_report_path, payload=step2_payload)
    scene_path = str(step2_payload.get("scene_path") or "")
    scene_dir = step3_root / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / "step3_report.json"

    log_dir = step3_root / LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scene_id}.log"

    cmd = build_worker_cmd(script_path=Path(__file__).resolve(), step2_report_path=step2_report_path, args=args)
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
        synthesize_step3_report_for_crash(
            report_path=report_path,
            scene_id=scene_id,
            scene_path=scene_path,
            step2_report=step2_report_path,
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
        synthesize_step3_report_for_crash(
            report_path=report_path,
            scene_id=scene_id,
            scene_path=scene_path,
            step2_report=step2_report_path,
            run_state=run_state,
            fail_reason=fail_reason,
            worker_exit_code=worker_exit_code,
            log_path=log_path,
            env_meta=env_meta,
        )

    parsed = parse_step3_report_summary(report_path)
    return {
        "idx": idx,
        "total": total,
        "scene_id": scene_id,
        "scene_path": scene_path,
        "step2_report": str(step2_report_path),
        "status": parsed["status"],
        "run_state": parsed["run_state"],
        "elapsed_sec": f"{elapsed:.3f}",
        "step3_ok": parsed["step3_ok"],
        "frames": parsed["frames"],
        "path_length_m": parsed["path_length_m"],
        "visited_unique_cells": parsed["visited_unique_cells"],
        "coverage_area_m2": parsed["coverage_area_m2"],
        "continuity_valid_ratio": parsed["continuity_valid_ratio"],
        "recovery_ratio": parsed["recovery_ratio"],
        "room_check_triggers_total": parsed["room_check_triggers_total"],
        "escape_attempts_total": parsed["escape_attempts_total"],
        "fail_reason": parsed["fail_reason"],
        "traj_meta_json": parsed["traj_meta_json"],
        "log_path": str(log_path),
        "worker_exit_code": worker_exit_code,
    }


def run_scene_inline(
    step2_report_path: Path,
    idx: int,
    total: int,
    step3_root: Path,
    env_meta: Dict,
    args,
) -> Dict:
    t0 = time.time()
    result = run_step3_scene_worker(step2_report_path=step2_report_path, step3_root=step3_root, args=args, env_meta=env_meta)
    elapsed = time.time() - t0
    return {
        "idx": idx,
        "total": total,
        "scene_id": result["scene_id"],
        "scene_path": result["scene_path"],
        "step2_report": str(step2_report_path),
        "status": result["status"],
        "run_state": result["run_state"],
        "elapsed_sec": f"{elapsed:.3f}",
        "step3_ok": result["step3_ok"],
        "frames": result["frames"],
        "path_length_m": result["path_length_m"],
        "visited_unique_cells": result["visited_unique_cells"],
        "coverage_area_m2": result["coverage_area_m2"],
        "continuity_valid_ratio": result["continuity_valid_ratio"],
        "recovery_ratio": result["recovery_ratio"],
        "room_check_triggers_total": result["room_check_triggers_total"],
        "escape_attempts_total": result["escape_attempts_total"],
        "fail_reason": result["fail_reason"],
        "traj_meta_json": result["traj_meta_json"],
        "log_path": "",
        "worker_exit_code": 0,
    }


def run_worker_entry(args):
    env_meta = collect_env_meta()
    step2_report = args.worker_step2_report.resolve()
    result = run_step3_scene_worker(
        step2_report_path=step2_report,
        step3_root=args.step3_root.resolve(),
        args=args,
        env_meta=env_meta,
    )
    print(
        f"[WORKER] {result['scene_id']} | status={result['status']} | run_state={result['run_state']} "
        f"| frames={result['frames']} | path={result['path_length_m']:.2f}m | coverage={result['coverage_area_m2']:.2f}m2 | fail={result['fail_reason']}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2-root", type=Path, default=DEFAULT_STEP2_ROOT)
    parser.add_argument("--step3-root", type=Path, default=None)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")

    parser.add_argument("--subprocess-isolation", dest="subprocess_isolation", action="store_true", default=True)
    parser.add_argument("--no-subprocess-isolation", dest="subprocess_isolation", action="store_false")
    parser.add_argument("--worker-timeout-sec", type=float, default=420.0)
    parser.add_argument("--worker-step2-report", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--hfov", type=float, default=0.0)
    parser.add_argument("--znear", type=float, default=0.0)
    parser.add_argument("--zfar", type=float, default=0.0)
    parser.add_argument("--disable-physics", action="store_true")

    parser.add_argument("--target-frames", type=int, default=240)
    parser.add_argument("--path-target-m", type=float, default=36.0)
    parser.add_argument("--path-max-m", type=float, default=44.0)
    parser.add_argument("--path-min-done-m", type=float, default=10.0)
    parser.add_argument("--grid-res-m", type=float, default=0.35)

    parser.add_argument("--wall-side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--wall-target-dist-m", type=float, default=0.55)
    parser.add_argument("--wall-search-radius-m", type=float, default=1.8)
    parser.add_argument("--wall-correction-gain", type=float, default=0.9)
    parser.add_argument("--wall-correction-max-m", type=float, default=0.35)
    parser.add_argument("--wall-open-penalty-m", type=float, default=1.35)

    parser.add_argument("--candidate-angle-step-deg", type=float, default=20.0)
    parser.add_argument("--candidate-max-turn-deg", type=float, default=120.0)
    parser.add_argument("--goal-probe-dists-m", type=float, action="append", default=None)
    parser.add_argument("--local-goal-min-m", type=float, default=0.45)
    parser.add_argument("--local-goal-max-m", type=float, default=2.2)
    parser.add_argument("--nav-step-m", type=float, default=0.22)
    parser.add_argument("--waypoint-reach-m", type=float, default=0.16)
    parser.add_argument("--same-island-only", dest="same_island_only", action="store_true", default=True)
    parser.add_argument("--no-same-island-only", dest="same_island_only", action="store_false")

    parser.add_argument("--blocked-delta-pos-m", type=float, default=0.03)
    parser.add_argument("--blocked-streak-replan", type=int, default=4)
    parser.add_argument("--no-plan-fail-streak", type=int, default=6)

    parser.add_argument("--poor-view-valid-ratio-min", type=float, default=0.45)
    parser.add_argument("--poor-view-black-ratio-max", type=float, default=0.40)
    parser.add_argument("--poor-view-replan-streak", type=int, default=2)
    parser.add_argument("--poor-view-fail-streak", type=int, default=10)
    parser.add_argument("--black-pixel-threshold", type=int, default=5)
    parser.add_argument("--depth-valid-min-m", type=float, default=1e-4)
    parser.add_argument("--depth-valid-max-m", type=float, default=200.0)
    parser.add_argument("--enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_true", default=False)
    parser.add_argument("--no-enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_false")
    parser.add_argument("--near-wall-depth-m", type=float, default=0.50)
    parser.add_argument("--blank-valid-ratio", type=float, default=0.05)
    parser.add_argument("--outside-valid-ratio-min", type=float, default=0.50)
    parser.add_argument("--outside-center-valid-ratio-max", type=float, default=0.08)
    parser.add_argument("--outside-side-valid-ratio-max", type=float, default=0.08)

    parser.add_argument("--recent-nav-window", type=int, default=40)
    parser.add_argument("--recent-overlap-radius-m", type=float, default=0.45)
    parser.add_argument("--coverage-plateau-window-frames", type=int, default=40)
    parser.add_argument("--coverage-plateau-new-cells-max", type=int, default=2)

    parser.add_argument("--camera-height-bias-m", type=float, default=0.08)
    parser.add_argument("--camera-height-min-m", type=float, default=1.58)
    parser.add_argument("--camera-height-max-m", type=float, default=1.78)
    parser.add_argument("--step3-pitch-deg", type=float, default=0.0)
    parser.add_argument("--max-yaw-step-deg", type=float, default=18.0)

    parser.add_argument("--origin-finish-pos-m", type=float, default=0.90)
    parser.add_argument("--origin-finish-yaw-deg", type=float, default=35.0)
    parser.add_argument("--origin-finish-streak", type=int, default=6)

    parser.add_argument("--export-frames", dest="export_frames", action="store_true", default=True)
    parser.add_argument("--no-export-frames", dest="export_frames", action="store_false")
    parser.add_argument("--make-video", dest="make_video", action="store_true", default=True)
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-fallback-gif", dest="video_fallback_gif", action="store_true", default=True)
    parser.add_argument("--no-video-fallback-gif", dest="video_fallback_gif", action="store_false")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.goal_probe_dists_m:
        args.goal_probe_dists_m = [0.8, 1.2, 1.6]
    else:
        args.goal_probe_dists_m = [float(v) for v in args.goal_probe_dists_m]

    args.step2_root = args.step2_root.resolve()
    if args.step3_root is None:
        args.step3_root = args.step2_root.parent / STEP3_DIRNAME
    args.step3_root = args.step3_root.resolve()
    args.step3_root.mkdir(parents=True, exist_ok=True)

    if args.worker_step2_report is not None:
        run_worker_entry(args)
        return

    step2_reports = discover_step2_reports(step2_root=args.step2_root, scene_id_filter=args.scene_id)
    if not step2_reports:
        print(f"No step2_floor_report.json found under: {args.step2_root}")
        return

    summary_path = args.step3_root / SUMMARY_PATH_NAME
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
    inventory_total = len(step2_reports)

    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        if write_header:
            writer.writeheader()
            f.flush()

        for loop_i, step2_report_path in enumerate(step2_reports, start=1):
            p2 = load_json(step2_report_path) or {}
            scene_id = scene_id_from_step2_report(report_path=step2_report_path, payload=p2)

            if scene_id in done_scene_ids:
                print(f"[{loop_i}/{inventory_total}] SKIP {scene_id} (already in summary)")
                continue
            if args.max_new > 0 and processed_new >= args.max_new:
                print(f"Reached --max-new={args.max_new}, stop.")
                break

            if args.subprocess_isolation:
                row = run_scene_with_subprocess(
                    step2_report_path=step2_report_path,
                    idx=loop_i,
                    total=inventory_total,
                    step3_root=args.step3_root,
                    env_meta=env_meta,
                    args=args,
                )
            else:
                row = run_scene_inline(
                    step2_report_path=step2_report_path,
                    idx=loop_i,
                    total=inventory_total,
                    step3_root=args.step3_root,
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
                "step2_report": row["step2_report"],
                "status": row["status"],
                "run_state": row["run_state"],
                "elapsed_sec": row["elapsed_sec"],
                "step3_ok": row["step3_ok"],
                "frames": row["frames"],
                "path_length_m": row["path_length_m"],
                "visited_unique_cells": row["visited_unique_cells"],
                "coverage_area_m2": row["coverage_area_m2"],
                "continuity_valid_ratio": row["continuity_valid_ratio"],
                "recovery_ratio": row["recovery_ratio"],
                "room_check_triggers_total": row["room_check_triggers_total"],
                "escape_attempts_total": row["escape_attempts_total"],
                "fail_reason": row["fail_reason"],
                "traj_meta_json": row["traj_meta_json"],
                "log_path": row["log_path"],
                "worker_exit_code": row["worker_exit_code"],
            }
            writer.writerow(row_out)
            f.flush()

            done_scene_ids.add(scene_id)
            processed_new += 1
            print(
                f"[{row['idx']}/{row['total']}] {scene_id} | {row['status']} | run_state={row['run_state']} "
                f"| frames={row['frames']} | path={row['path_length_m']:.2f}m | coverage={row['coverage_area_m2']:.2f}m2 "
                f"| continuity={row['continuity_valid_ratio']:.3f} | fail={row['fail_reason']}"
            )

    print(
        f"Step3 navi batch done: inventory_total={inventory_total}, newly_processed={processed_new}, "
        f"resume={'on' if args.resume else 'off'}, subprocess_isolation={'on' if args.subprocess_isolation else 'off'}"
    )


if __name__ == "__main__":
    main()
