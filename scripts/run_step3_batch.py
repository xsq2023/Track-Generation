#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg

ROOT = Path("/Users/sota/code/3d_new")
DEFAULT_OUTPUT_ROOT = ROOT / "output"
DEFAULT_STEP2_ROOT = DEFAULT_OUTPUT_ROOT / "step2"
STEP3_DIRNAME = "step3"
SUMMARY_PATH_NAME = "_batch_summary.tsv"
LOG_DIRNAME = "_batch_logs"

SUMMARY_FIELDS = [
    "row_id",
    "idx",
    "total",
    "scene_id",
    "scene_path",
    "step2_report",
    "status",
    "run_state",
    "elapsed_sec",
    "step3_ok",
    "frames",
    "path_length_m",
    "visited_unique_cells",
    "coverage_area_m2",
    "continuity_valid_ratio",
    "recovery_ratio",
    "room_check_triggers_total",
    "escape_attempts_total",
    "fail_reason",
    "traj_meta_json",
    "log_path",
    "worker_exit_code",
]

ALLOWED_FLOOR_STATUS = {"OK", "FALLBACK_OK", "SLOPE_OK"}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_angle_rad(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff_rad(target: float, current: float) -> float:
    return wrap_angle_rad(target - current)


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


def append_jsonl(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=to_json_compatible))
        f.write("\n")


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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


def scene_id_from_step2_report(report_path: Path, payload: Optional[Dict]) -> str:
    if isinstance(payload, dict):
        sid = payload.get("scene_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
        sp = payload.get("scene_path")
        if isinstance(sp, str) and sp.strip():
            stem = Path(sp).stem
            cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem).strip("_")
            if cleaned:
                return cleaned
    return report_path.parent.name


def resolve_sensor_config(step2_payload: Dict, args) -> Dict:
    base = step2_payload.get("sensor_config") if isinstance(step2_payload.get("sensor_config"), dict) else {}
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


def extract_start_pose(step2_payload: Dict, args) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[str]]:
    floor_pose = step2_payload.get("floor_pose_apply_readback_after") if isinstance(step2_payload.get("floor_pose_apply_readback_after"), dict) else {}
    if isinstance(floor_pose.get("position"), list) and len(floor_pose.get("position")) >= 3:
        p = np.asarray(floor_pose["position"][:3], dtype=np.float64)
        yaw = float(floor_pose.get("yaw_rad", 0.0))
        pitch = math.radians(float(args.step3_pitch_deg))
        return p, yaw, pitch, None

    step1_report = step2_payload.get("step1_report")
    if isinstance(step1_report, str) and step1_report.strip():
        p1 = load_json(Path(step1_report))
        if isinstance(p1, dict):
            pose = p1.get("chosen_start_pose") if isinstance(p1.get("chosen_start_pose"), dict) else {}
            if isinstance(pose.get("position"), list) and len(pose.get("position")) >= 3:
                p = np.asarray(pose["position"][:3], dtype=np.float64)
                yaw = float(pose.get("yaw_rad", 0.0))
                pitch = math.radians(float(args.step3_pitch_deg))
                return p, yaw, pitch, None

    return None, None, None, "start_pose_missing"


def dependency_gate(step2_payload: Dict) -> Tuple[bool, str]:
    if not bool(step2_payload.get("step2_ok", False)):
        return False, "step2_not_ok"
    floor_status = str(step2_payload.get("floor_status") or "")
    if floor_status not in ALLOWED_FLOOR_STATUS:
        return False, "step2_not_ok"
    if step2_payload.get("cam_y_minus_floor_height") is None:
        return False, "floor_height_missing"
    return True, "OK"


def percentile_or_default(values: np.ndarray, q: float, default: float) -> float:
    if values.size <= 0:
        return float(default)
    return float(np.percentile(values, q))


def depth_sector_values(depth: np.ndarray, valid: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> Tuple[np.ndarray, float]:
    h, w = depth.shape
    ry0 = int(np.clip(round(h * y0), 0, h - 1))
    ry1 = int(np.clip(round(h * y1), ry0 + 1, h))
    rx0 = int(np.clip(round(w * x0), 0, w - 1))
    rx1 = int(np.clip(round(w * x1), rx0 + 1, w))
    block = depth[ry0:ry1, rx0:rx1]
    block_valid = valid[ry0:ry1, rx0:rx1]
    total = int(block.size)
    if total <= 0 or (not np.any(block_valid)):
        return np.zeros((0,), dtype=np.float32), 0.0
    vals = block[block_valid]
    return vals, float(vals.size / total)


def depth_proxy_stats(depth: np.ndarray, args) -> Dict:
    valid = np.isfinite(depth) & (depth > float(args.depth_valid_min_m))
    if bool(args.enforce_depth_valid_max):
        valid &= depth < float(args.depth_valid_max_m)

    valid_ratio = float(np.mean(valid))
    center, center_valid_ratio = depth_sector_values(depth, valid, 0.45, 0.90, 0.35, 0.65)
    left, left_valid_ratio = depth_sector_values(depth, valid, 0.45, 0.90, 0.00, 0.35)
    right, right_valid_ratio = depth_sector_values(depth, valid, 0.45, 0.90, 0.65, 1.00)
    fwd_left, fwd_left_valid_ratio = depth_sector_values(depth, valid, 0.45, 0.90, 0.20, 0.55)

    clearance_fwd = percentile_or_default(center, 30.0, 0.0)
    clearance_left = percentile_or_default(left, 30.0, 0.0)
    clearance_right = percentile_or_default(right, 30.0, 0.0)
    clearance_fwd_left = percentile_or_default(fwd_left, 30.0, 0.0)
    min_depth = percentile_or_default(center, 5.0, 0.0)

    near_wall_ratio = 0.0
    if center.size > 0:
        near_wall_ratio = float(np.mean(center < float(args.near_wall_depth_m)))

    outside_proxy = bool(
        valid_ratio < float(args.outside_valid_ratio_min)
        or (
            center_valid_ratio < float(args.outside_center_valid_ratio_max)
            and left_valid_ratio < float(args.outside_side_valid_ratio_max)
            and right_valid_ratio < float(args.outside_side_valid_ratio_max)
        )
    )

    return {
        "valid_ratio": valid_ratio,
        "center_valid_ratio": center_valid_ratio,
        "left_valid_ratio": left_valid_ratio,
        "right_valid_ratio": right_valid_ratio,
        "fwd_left_valid_ratio": fwd_left_valid_ratio,
        "clearance_fwd_m": clearance_fwd,
        "clearance_left_m": clearance_left,
        "clearance_right_m": clearance_right,
        "clearance_fwd_left_m": clearance_fwd_left,
        "min_depth_m": min_depth,
        "near_wall_ratio": near_wall_ratio,
        "outside_proxy": outside_proxy,
        "blank_frame": bool(valid_ratio < float(args.blank_valid_ratio)),
    }


def grid_cell(pos: np.ndarray, grid_res_m: float) -> Tuple[int, int]:
    return (int(math.floor(float(pos[0]) / grid_res_m)), int(math.floor(float(pos[2]) / grid_res_m)))


def sample_cells_along_heading(
    pos: np.ndarray,
    yaw_rad: float,
    sample_len_m: float,
    sample_step_m: float,
    grid_res_m: float,
) -> List[Tuple[int, int]]:
    if sample_len_m <= 0.0:
        return []
    n = int(max(1, math.floor(sample_len_m / max(sample_step_m, 1e-3))))
    out = []
    for i in range(1, n + 1):
        d = float(i) * sample_step_m
        x = float(pos[0]) + math.sin(yaw_rad) * d
        z = float(pos[2]) - math.cos(yaw_rad) * d
        out.append((int(math.floor(x / grid_res_m)), int(math.floor(z / grid_res_m))))
    return out


def novelty_and_frontier(visited: set, pos: np.ndarray, yaw_rad: float, clearance_m: float, args) -> Tuple[float, float]:
    sample_len = min(float(clearance_m), float(args.novelty_cap_m))
    cells = sample_cells_along_heading(
        pos=pos,
        yaw_rad=yaw_rad,
        sample_len_m=sample_len,
        sample_step_m=float(args.novelty_sample_step_m),
        grid_res_m=float(args.grid_res_m),
    )
    if not cells:
        return 0.0, 0.0

    nov = [1.0 if c not in visited else 0.0 for c in cells]
    novelty = float(sum(nov) / len(nov))

    frontier_hits = 0
    prev_visited = cells[0] in visited
    for c in cells[1:]:
        cur_visited = c in visited
        if prev_visited and (not cur_visited):
            frontier_hits += 1
        prev_visited = cur_visited
    frontier_norm = float(frontier_hits / max(len(cells) - 1, 1))
    return novelty, frontier_norm


def classify_geom_event(stats: Dict, delta_l: float, delta_f: float, args) -> str:
    fwd = float(stats["clearance_fwd_m"])
    left = float(stats["clearance_left_m"])
    fwd_left = float(stats["clearance_fwd_left_m"])
    right = float(stats["clearance_right_m"])
    near_wall_ratio = float(stats["near_wall_ratio"])
    valid_ratio = float(stats["valid_ratio"])

    votes = 0
    if delta_l > float(args.opening_delta_l_m):
        votes += 1
    if left > float(args.opening_left_m):
        votes += 1
    if fwd > float(args.opening_fwd_m):
        votes += 1

    opening = False
    if valid_ratio >= float(args.opening_min_valid_ratio):
        if not (bool(args.opening_require_right_cap) and right >= float(args.opening_right_cap_m)):
            opening = votes >= int(args.opening_votes_needed)

    corner = (
        (fwd < float(args.corner_strong_f_m) and fwd_left < float(args.corner_strong_fl_m))
        or (delta_f < float(args.corner_delta_f_m) and fwd_left < float(args.corner_delta_fl_m))
        or (fwd < float(args.corner_near_f_m) and near_wall_ratio > float(args.near_wall_ratio_high))
    )
    if opening and corner:
        if delta_l >= float(args.opening_override_delta_l_m) and fwd >= float(args.opening_override_fwd_m):
            return "OPENING_OUTER"
        return "CORNER_INNER"
    if opening:
        return "OPENING_OUTER"
    if corner:
        return "CORNER_INNER"
    return "NONE"


def make_step3_debug_strip(images: List[np.ndarray], out_path: Path):
    if not images:
        return
    thumbs = []
    for img in images:
        pil = Image.fromarray(img)
        pil = pil.resize((160, 120), Image.Resampling.BILINEAR)
        thumbs.append(np.asarray(pil, dtype=np.uint8))
    strip = np.concatenate(thumbs, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path)


def make_visited_map(visited: set, out_path: Path):
    if not visited:
        return
    xs = [c[0] for c in visited]
    zs = [c[1] for c in visited]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    w = max(1, max_x - min_x + 1)
    h = max(1, max_z - min_z + 1)
    canvas = np.zeros((h, w), dtype=np.uint8)
    for cx, cz in visited:
        x = cx - min_x
        z = cz - min_z
        canvas[h - 1 - z, x] = 255
    Image.fromarray(canvas).save(out_path)


def encode_video_from_frames(frames_dir: Path, out_path: Path, fps: int) -> Tuple[bool, str]:
    if not frames_dir.exists():
        return False, "video_encode_skipped_frames_dir_missing"
    if not any(frames_dir.glob("frame_*.png")):
        return False, "video_encode_skipped_no_frames"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(int(max(1, fps))),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return False, "video_encode_skipped_ffmpeg_missing"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        if len(err) > 300:
            err = err[-300:]
        return False, f"video_encode_failed: {err}"
    return True, "ok"


def encode_gif_from_frames(frames_dir: Path, out_path: Path, fps: int) -> Tuple[bool, str]:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        return False, "gif_encode_skipped_no_frames"
    frames = []
    for p in frame_paths:
        try:
            with Image.open(p) as im:
                frames.append(im.convert("RGB"))
        except Exception:
            continue
    if not frames:
        return False, "gif_encode_skipped_unreadable_frames"
    duration_ms = int(max(20, round(1000.0 / max(1, int(fps)))))
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    except Exception as exc:
        return False, f"gif_encode_failed: {exc}"
    return True, "ok"


def try_load_navmesh_cache(sim: habitat_sim.Simulator, cache_path: Path) -> Dict:
    if not cache_path.exists():
        return {
            "loaded": False,
            "source": "cache_missing",
            "cache_path": str(cache_path),
            "error": "navmesh_cache_not_found",
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


def try_load_navmesh_from_step2(sim: habitat_sim.Simulator, step2_payload: Dict) -> Dict:
    step1_report_raw = step2_payload.get("step1_report")
    if not isinstance(step1_report_raw, str) or not step1_report_raw.strip():
        return {
            "loaded": False,
            "source": "step2_missing_step1_report",
            "cache_path": None,
            "error": "step2_step1_report_missing",
        }

    step1_payload = load_json(Path(step1_report_raw))
    if not isinstance(step1_payload, dict):
        return {
            "loaded": False,
            "source": "step1_report_invalid",
            "cache_path": None,
            "error": "step1_report_missing_or_invalid",
        }

    navmesh_probe = step1_payload.get("navmesh_probe") if isinstance(step1_payload.get("navmesh_probe"), dict) else {}
    cache_path_raw = navmesh_probe.get("cache_path")
    if isinstance(cache_path_raw, str) and cache_path_raw.strip():
        return try_load_navmesh_cache(sim=sim, cache_path=Path(cache_path_raw))

    step0_scene_init_raw = step1_payload.get("step0_scene_init")
    if not isinstance(step0_scene_init_raw, str) or not step0_scene_init_raw.strip():
        return {
            "loaded": False,
            "source": "step1_missing_step0_scene_init",
            "cache_path": None,
            "error": "step1_step0_scene_init_missing",
        }

    step0_payload = load_json(Path(step0_scene_init_raw))
    if not isinstance(step0_payload, dict):
        return {
            "loaded": False,
            "source": "step0_scene_init_invalid",
            "cache_path": None,
            "error": "step0_scene_init_missing_or_invalid",
        }

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
            "source": "step0_cache_missing",
            "cache_path": None,
            "error": "step0_navmesh_cache_missing",
        }
    return try_load_navmesh_cache(sim=sim, cache_path=Path(cache_path_raw))


def try_navmesh_step(pathfinder, start: np.ndarray, target: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if pathfinder is None or (not bool(pathfinder.is_loaded)):
        return None, "navmesh_not_loaded"
    try:
        stepped = pathfinder.try_step(np.asarray(start, dtype=np.float32), np.asarray(target, dtype=np.float32))
        stepped_np = np.array([float(stepped[0]), float(stepped[1]), float(stepped[2])], dtype=np.float64)
        if not np.all(np.isfinite(stepped_np)):
            return None, "navmesh_try_step_nonfinite"
        return stepped_np, None
    except Exception as exc:
        return None, f"navmesh_try_step_exception:{exc}"


def choose_turn_commit_sign(left: float, right: float, delta_l: float, delta_f: float) -> float:
    if left > right + 0.15:
        return 1.0
    if right > left + 0.15:
        return -1.0
    if delta_l > 0.10:
        return 1.0
    if delta_l < -0.10:
        return -1.0
    if delta_f < -0.15:
        return 1.0 if left >= right else -1.0
    return 1.0 if left >= right else -1.0


def compute_step_length(fwd: float, state: str, args) -> float:
    if state == "ROOM_CHECK_COMMIT":
        step_nominal = float(args.step_nominal_commit_m)
        step_scale = float(args.step_clearance_scale_commit)
    elif state == "ESCAPE":
        step_nominal = float(args.step_nominal_escape_m)
        step_scale = float(args.step_clearance_scale_escape)
    else:
        step_nominal = float(args.step_nominal_wall_m)
        step_scale = float(args.step_clearance_scale_wall)
    step_min = float(args.step_min_m)
    step_max = float(args.step_max_m)
    step_hi = max(step_min, min(step_max, step_scale * max(fwd, 0.0)))
    step = clamp(step_nominal, step_min, step_hi)
    if fwd > float(args.anti_freeze_fwd_m):
        step = min(step_max, max(step, float(args.anti_freeze_step_m)))
    if state == "TURN_IN_PLACE":
        return 0.0
    if fwd < float(args.step_block_fwd_m):
        return 0.0
    return step


def scan_headings(
    sim: habitat_sim.Simulator,
    agent,
    pos: np.ndarray,
    yaw_ref: float,
    pitch_rad: float,
    visited: set,
    forbid_yaw_rad: Optional[float],
    forbid_half_angle_rad: float,
    args,
) -> Dict:
    bins = int(args.escape_scan_bins)
    rows = []
    for i in range(bins):
        yaw = wrap_angle_rad(yaw_ref + 2.0 * math.pi * float(i) / float(bins))
        obs = observe_pose(sim=sim, agent=agent, position=pos.astype(np.float32), yaw_rad=yaw, pitch_rad=pitch_rad)
        stats = depth_proxy_stats(np.asarray(obs["depth"], dtype=np.float32), args=args)
        nov, frontier = novelty_and_frontier(visited=visited, pos=pos, yaw_rad=yaw, clearance_m=stats["clearance_fwd_m"], args=args)
        score = float(stats["clearance_fwd_m"]) * (
            1.0 + float(args.lambda_novelty) * nov + float(args.lambda_frontier) * frontier
        )
        forbidden = False
        if forbid_yaw_rad is not None and abs(angle_diff_rad(yaw, float(forbid_yaw_rad))) <= float(forbid_half_angle_rad):
            forbidden = True
            score = score * float(args.no_back_forbid_score_scale)
        rows.append(
            {
                "yaw_rad": yaw,
                "yaw_deg": float(math.degrees(yaw)),
                "score": score,
                "clearance": float(stats["clearance_fwd_m"]),
                "novelty": nov,
                "frontier_hits_norm": frontier,
                "forbidden": bool(forbidden),
            }
        )

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    usable = [r for r in rows if not bool(r.get("forbidden", False))]
    ranked = usable if usable else rows
    top_k = ranked[: max(1, int(args.escape_scan_top_k))]
    return {
        "ranked": ranked,
        "top": top_k,
        "best": (top_k[0] if top_k else None),
    }


def choose_room_check_heading(
    sim: habitat_sim.Simulator,
    agent,
    pos: np.ndarray,
    yaw_ref: float,
    pitch_rad: float,
    visited: set,
    left: float,
    right: float,
    args,
) -> Dict:
    preferred_sign = 1.0 if left >= right else -1.0
    scan = scan_headings(
        sim=sim,
        agent=agent,
        pos=pos,
        yaw_ref=yaw_ref,
        pitch_rad=pitch_rad,
        visited=visited,
        forbid_yaw_rad=None,
        forbid_half_angle_rad=0.0,
        args=args,
    )
    best_row = None
    best_score = None
    for row in scan.get("ranked") or []:
        diff = angle_diff_rad(float(row["yaw_rad"]), float(yaw_ref))
        if abs(diff) > math.radians(105.0):
            continue
        side_bias = 0.0
        if preferred_sign * diff > math.radians(20.0):
            side_bias += 0.35
        if preferred_sign * diff > math.radians(45.0):
            side_bias += 0.25
        if abs(diff) < math.radians(30.0):
            side_bias -= 0.20
        score = float(row["score"]) * (1.0 + side_bias)
        if best_row is None or score > float(best_score):
            best_row = row
            best_score = score
    if best_row is None:
        fallback_yaw = wrap_angle_rad(float(yaw_ref) + preferred_sign * math.radians(65.0))
        return {
            "yaw_rad": fallback_yaw,
            "yaw_deg": float(math.degrees(fallback_yaw)),
            "score": None,
            "source": "fallback_offset",
        }
    return {
        "yaw_rad": float(best_row["yaw_rad"]),
        "yaw_deg": float(best_row["yaw_deg"]),
        "score": float(best_score),
        "source": "opening_scan",
    }


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
    floor_height = float(step2_payload.get("floor_height") if step2_payload.get("floor_height") is not None else 0.0)

    report = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 3,
        "step_name": "step3_explorer_fsm",
        "step2_report": str(step2_report_path),
        "scene_source": str(scene_source),
        "sensor_config": sensor_cfg,
        "start_pose": {
            "position": start_pos.tolist(),
            "yaw_rad": float(start_yaw),
            "pitch_rad": float(start_pitch),
        },
        "environment": env_meta,
        "params": {
            "target_frames": int(args.target_frames),
            "path_min_m": float(args.path_min_done_m),
            "path_max_m": float(args.path_max_m),
            "plateau_window_path_m": float(args.plateau_window_path_m),
            "plateau_new_cells_max": int(args.plateau_new_cells_max),
            "startup_wall_frames": int(args.startup_wall_frames),
            "startup_wall_path_m": float(args.startup_wall_path_m),
            "opening_persist_frames": int(args.opening_persist_frames),
            "black_soft_void_ratio": float(args.black_soft_void_ratio),
            "black_soft_ratio": float(args.black_soft_ratio),
            "grid_res_m": float(args.grid_res_m),
            "state_priority": "safety>turn>escape>room_check>wall_follow",
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
    collision_frames = 0
    blank_streak = 0
    black_high_streak = 0
    black_soft_streak = 0
    max_black_soft_streak = 0
    black_soft_alert_frames = 0
    depth_low_streak = 0
    outside_proxy_streak = 0
    black_alert_frames = 0
    outside_proxy_frames = 0
    inside_reanchor_count = 0
    zero_move_streak = 0
    max_zero_move_streak = 0
    coverage_stall_streak = 0
    max_coverage_stall_streak = 0

    room_check_triggers_total = 0
    room_check_success_total = 0
    escape_attempts_total = 0
    escape_success_total = 0
    escape_attempts_consecutive = 0

    fsm_state = "WALL_FOLLOW"
    fsm_prev_state = "WALL_FOLLOW"
    fsm_state_frames = 0

    heading_lock: Optional[float] = None
    turn_commit_sign: Optional[float] = None
    commit_dist_left_m = 0.0
    commit_frames_left = 0
    room_check_start_cells = 0
    escape_start_cells = 0
    escape_recovery_frames_left = 0
    escape_recovery_mode = "none"

    last_escape_frame = -100000
    max_consecutive_recovery = 0
    consecutive_recovery = 0

    history_l = deque(maxlen=10)
    history_f = deque(maxlen=10)
    wall_lock_recent = deque(maxlen=8)
    window = deque(maxlen=int(args.stuck_window_frames))
    plateau_window = deque()
    corner_lock_level = 0

    fail_reason_hist = Counter()
    debug_images = []
    warnings = []

    visited = set()
    traj_records = 0

    pos = start_pos.copy()
    yaw = float(start_yaw)
    pitch = float(start_pitch)

    origin_pos = start_pos.copy()
    origin_yaw = float(start_yaw)
    origin_match_streak = 0
    coverage_gain_recent = 0
    coverage_gain_last_path_m = 0
    net_disp_last_path_m = 0.0
    plateau_reason = None
    sum_black_void_ratio = 0.0
    wall_lock_frames = 0
    startup_wall_guard_frames = 0
    opening_persist_count = 0
    max_opening_persist_count = 0
    last_wall_lock = False
    last_startup_wall_guard_active = False
    last_effective_move_yaw = float(start_yaw)
    no_back_yaw_rad = None
    no_back_until_frame = -1
    escape_blocked_streak = 0
    max_escape_blocked_streak = 0
    navmesh_probe = {
        "loaded": False,
        "source": "not_attempted",
        "cache_path": None,
        "error": None,
    }
    use_navmesh_move = False

    terminal_record = {
        "last_fsm_state": fsm_state,
        "gate_name": None,
        "fail_reason": None,
        "counters_snapshot": {},
    }

    try:
        ts = time.time()
        scene_seed = int((int(args.seed) + (abs(hash(scene_id)) % 2147483647)) % 2147483647)
        sim = build_sim(scene_source=scene_source, sensor_cfg=sensor_cfg, seed=scene_seed, enable_physics=(not args.disable_physics))
        agent = sim.initialize_agent(0)
        timings["t_load_scene"] = float(time.time() - ts)
        ts = time.time()
        navmesh_probe = try_load_navmesh_from_step2(sim=sim, step2_payload=step2_payload)
        use_navmesh_move = bool(navmesh_probe.get("loaded", False))
        timings["t_load_navmesh"] = float(time.time() - ts)
        report["navmesh_probe"] = navmesh_probe
        report["use_navmesh_move"] = bool(use_navmesh_move)

        done_reason = None

        max_frames_guard = int(max(args.target_frames * 3, args.target_frames + 40))
        for frame_idx in range(max_frames_guard):
            frame_t0 = time.time()

            ts_render = time.time()
            obs = observe_pose(sim=sim, agent=agent, position=pos.astype(np.float32), yaw_rad=yaw, pitch_rad=pitch)
            rgb = np.asarray(obs["rgb"], dtype=np.uint8)
            depth = np.asarray(obs["depth"], dtype=np.float32)
            pose_readback = obs["pose_readback"]
            pos = np.asarray(pose_readback["position"], dtype=np.float64)
            timings_render = float(time.time() - ts_render)

            if bool(args.export_frames):
                Image.fromarray(rgb).save(frames_dir / f"frame_{frame_idx:04d}.png")

            if frame_idx % 10 == 0:
                debug_images.append(rgb.copy())

            stats = depth_proxy_stats(depth=depth, args=args)
            fwd = float(stats["clearance_fwd_m"])
            left = float(stats["clearance_left_m"])
            right = float(stats["clearance_right_m"])
            fwd_left = float(stats["clearance_fwd_left_m"])
            min_depth = float(stats["min_depth_m"])
            near_wall_ratio = float(stats["near_wall_ratio"])
            valid_ratio = float(stats["valid_ratio"])
            outside_proxy = bool(stats["outside_proxy"])
            black_mask = np.all(rgb <= int(args.black_pixel_threshold), axis=2)
            black_ratio = float(np.mean(black_mask))
            depth_valid_mask = np.isfinite(depth) & (depth > float(args.depth_valid_min_m))
            if bool(args.enforce_depth_valid_max):
                depth_valid_mask &= depth < float(args.depth_valid_max_m)
            black_void_ratio = float(np.mean(black_mask & (~depth_valid_mask)))
            sum_black_void_ratio += black_void_ratio

            black_alert = bool(black_ratio > float(args.black_ratio_recovery))
            if black_alert:
                black_high_streak += 1
                black_alert_frames += 1
            else:
                black_high_streak = 0

            black_soft_alert = bool(
                black_void_ratio > float(args.black_soft_void_ratio)
                or (black_ratio > float(args.black_soft_ratio) and valid_ratio < 0.75)
            )
            if black_soft_alert:
                black_soft_streak += 1
                black_soft_alert_frames += 1
            else:
                black_soft_streak = 0
            max_black_soft_streak = max(max_black_soft_streak, int(black_soft_streak))

            if valid_ratio < float(args.depth_valid_ratio_recovery_min):
                depth_low_streak += 1
            else:
                depth_low_streak = 0

            if outside_proxy:
                outside_proxy_streak += 1
                outside_proxy_frames += 1
            else:
                outside_proxy_streak = 0

            if bool(stats["blank_frame"]):
                blank_streak += 1
            else:
                blank_streak = 0

            if blank_streak >= int(args.blank_frame_streak_max):
                fail_reason = "blank_frame_streak"
                fail_reason_detail = f"blank_streak={blank_streak} >= {int(args.blank_frame_streak_max)}"
                done_reason = "FAIL"
                terminal_record.update(
                    {
                        "last_fsm_state": fsm_state,
                        "gate_name": "blank_frame_guard",
                        "fail_reason": fail_reason,
                    }
                )
                break

            if depth_low_streak >= int(args.depth_valid_ratio_fail_streak):
                fail_reason = "outside_scene_or_invalid_depth"
                fail_reason_detail = (
                    f"valid_ratio={valid_ratio:.3f}, depth_low_streak={depth_low_streak} "
                    f">= depth_valid_ratio_fail_streak={int(args.depth_valid_ratio_fail_streak)}"
                )
                if (
                    path_length_m >= float(args.path_min_done_m)
                    or path_length_m >= float(args.hazard_soft_timeout_path_m)
                ):
                    done_reason = "TIMEOUT"
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "depth_gate_soft_timeout",
                            "fail_reason": fail_reason,
                        }
                    )
                else:
                    done_reason = "FAIL"
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "depth_gate_hard_fail",
                            "fail_reason": fail_reason,
                        }
                    )
                break

            black_soft_recovery = bool(black_soft_streak >= 2)
            hard_recovery_trigger = bool(
                depth_low_streak >= int(args.depth_valid_ratio_recovery_streak)
                or outside_proxy_streak >= int(args.outside_proxy_recovery_streak)
            )
            inside_recovery_trigger = bool(hard_recovery_trigger)
            reanchor_recovery_trigger = bool(hard_recovery_trigger)
            inside_force_reanchor = False
            if (
                reanchor_recovery_trigger
                and bool(args.enable_inside_reanchor)
                and outside_proxy_streak >= int(args.outside_proxy_reanchor_streak)
                and inside_reanchor_count < int(args.inside_reanchor_max)
            ):
                inside_force_reanchor = True
                inside_reanchor_count += 1

            if outside_proxy_streak >= int(args.outside_proxy_fail_streak):
                fail_reason = "outside_scene_or_blackframe"
                fail_reason_detail = (
                    f"outside_proxy_streak={outside_proxy_streak} >= outside_proxy_fail_streak={int(args.outside_proxy_fail_streak)}"
                )
                if (
                    path_length_m >= float(args.path_min_done_m)
                    or path_length_m >= float(args.hazard_soft_timeout_path_m)
                ):
                    done_reason = "TIMEOUT"
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "outside_proxy_soft_timeout",
                            "fail_reason": fail_reason,
                        }
                    )
                else:
                    done_reason = "FAIL"
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "outside_proxy_guard",
                            "fail_reason": fail_reason,
                        }
                    )
                break

            black_hazard = bool(black_soft_streak >= 6 and depth_low_streak >= 3)
            force_escape_from_black = bool(black_hazard and frame_idx >= 5 and fsm_state != "ESCAPE")

            cur_cell = grid_cell(pos=pos, grid_res_m=float(args.grid_res_m))
            visited_before = len(visited)
            visited.add(cur_cell)
            visited_unique_cells = len(visited)
            coverage_area_m2 = float(visited_unique_cells) * float(args.grid_res_m) * float(args.grid_res_m)
            new_cell = 1 if visited_unique_cells > visited_before else 0

            l_base = float(np.median(np.asarray(history_l, dtype=np.float64))) if len(history_l) >= 3 else left
            f_base = float(np.median(np.asarray(history_f, dtype=np.float64))) if len(history_f) >= 3 else fwd
            delta_l = float(left - l_base)
            delta_f = float(fwd - f_base)
            geom_event = classify_geom_event(stats=stats, delta_l=delta_l, delta_f=delta_f, args=args)

            wall_lock_recent.append(1 if (0.25 <= left <= 1.2) else 0)
            wall_lock = bool(len(wall_lock_recent) >= 8 and sum(wall_lock_recent) >= 6)
            last_wall_lock = wall_lock
            if wall_lock:
                wall_lock_frames += 1

            if geom_event == "OPENING_OUTER":
                opening_persist_count += 1
            else:
                opening_persist_count = 0
            max_opening_persist_count = max(max_opening_persist_count, int(opening_persist_count))

            startup_wall_guard_active = bool(
                frame_idx < int(args.startup_wall_frames)
                and path_length_m < float(args.startup_wall_path_m)
            )
            last_startup_wall_guard_active = startup_wall_guard_active
            if startup_wall_guard_active:
                startup_wall_guard_frames += 1

            novelty_score, frontier_hits_norm = novelty_and_frontier(
                visited=visited,
                pos=pos,
                yaw_rad=yaw,
                clearance_m=fwd,
                args=args,
            )

            history_l.append(left)
            history_f.append(fwd)

            window.append(
                {
                    "pos": pos.copy(),
                    "path": path_length_m,
                    "visited": visited_unique_cells,
                    "near_wall_ratio": near_wall_ratio,
                }
            )

            stuck = False
            net_disp_window = 0.0
            path_window = 0.0
            new_cells_window = 0
            if len(window) >= int(args.stuck_window_frames):
                oldest = window[0]
                net_disp_window = float(np.linalg.norm(pos - oldest["pos"]))
                path_window = float(path_length_m - oldest["path"])
                new_cells_window = int(visited_unique_cells - oldest["visited"])
                near_wall_high_ratio = float(
                    np.mean([1.0 if w["near_wall_ratio"] >= float(args.near_wall_ratio_high) else 0.0 for w in window])
                )
                stuck = bool(
                    net_disp_window < float(args.stuck_net_disp_m)
                    or path_window < float(args.stuck_path_m)
                    or new_cells_window <= int(args.stuck_new_cells_max)
                    or near_wall_high_ratio >= float(args.stuck_near_wall_ratio)
                )

            coverage_gain_recent = max(0, new_cells_window)
            if len(window) >= int(args.stuck_window_frames) and new_cells_window <= int(args.coverage_stall_new_cells_max):
                coverage_stall_streak += 1
            else:
                coverage_stall_streak = 0
            max_coverage_stall_streak = max(max_coverage_stall_streak, coverage_stall_streak)
            if zero_move_streak >= int(args.zero_move_escape_streak) or coverage_stall_streak >= int(args.coverage_stall_escape_streak):
                stuck = True

            # FSM transition
            fsm_prev_state = fsm_state
            room_check_trigger = False
            scan_phase = "none"
            scan_score_best = None
            scan_best_heading_deg = None
            turn_locked = False

            boundary_escape_trigger = bool(
                frame_idx >= 5
                and fsm_state != "ESCAPE"
                and (
                    black_soft_alert
                    or hard_recovery_trigger
                    or (outside_proxy and black_alert)
                )
            )
            safety_turn_trigger = bool(min_depth > 0 and min_depth < float(args.safety_min_depth_m))

            if boundary_escape_trigger:
                fsm_state = "ESCAPE"
                fsm_state_frames = 0
                heading_lock = None
                turn_commit_sign = None
                escape_recovery_frames_left = max(int(escape_recovery_frames_left), 2)
                escape_recovery_mode = "backoff"
            elif safety_turn_trigger and fsm_state not in {"TURN_IN_PLACE", "ESCAPE"}:
                fsm_state = "TURN_IN_PLACE"
                fsm_state_frames = 0
                corner_lock_level = 0
                turn_commit_sign = choose_turn_commit_sign(left=left, right=right, delta_l=delta_l, delta_f=delta_f)
            elif force_escape_from_black:
                fsm_state = "ESCAPE"
                fsm_state_frames = 0
                heading_lock = None
                turn_commit_sign = None
                escape_recovery_frames_left = max(int(escape_recovery_frames_left), 2)
                escape_recovery_mode = "backoff"
            elif fsm_state == "WALL_FOLLOW":
                allow_explore = not bool(startup_wall_guard_active)
                if (
                    allow_explore
                    and (
                        zero_move_streak >= int(args.zero_move_escape_streak)
                        or coverage_stall_streak >= int(args.coverage_stall_escape_streak)
                    )
                    and (frame_idx - last_escape_frame) >= int(args.escape_cooldown_frames)
                ):
                    fsm_state = "ESCAPE"
                    fsm_state_frames = 0
                    heading_lock = None
                    turn_commit_sign = None
                elif geom_event == "CORNER_INNER" or fwd < float(args.turn_enter_fwd_m):
                    fsm_state = "TURN_IN_PLACE"
                    fsm_state_frames = 0
                    corner_lock_level = 0
                    turn_commit_sign = choose_turn_commit_sign(left=left, right=right, delta_l=delta_l, delta_f=delta_f)
                elif (
                    allow_explore
                    and geom_event == "OPENING_OUTER"
                    and opening_persist_count >= int(args.opening_persist_frames)
                ):
                    opening_heading = choose_room_check_heading(
                        sim=sim,
                        agent=agent,
                        pos=pos,
                        yaw_ref=yaw,
                        pitch_rad=pitch,
                        visited=visited,
                        left=left,
                        right=right,
                        args=args,
                    )
                    fsm_state = "ROOM_CHECK_COMMIT"
                    fsm_state_frames = 0
                    room_check_trigger = True
                    room_check_triggers_total += 1
                    heading_lock = float(opening_heading["yaw_rad"])
                    scan_best_heading_deg = float(opening_heading["yaw_deg"])
                    scan_score_best = opening_heading["score"]
                    commit_dist_left_m = float(args.room_check_commit_dist_m)
                    commit_frames_left = int(args.commit_frames_max)
                    room_check_start_cells = visited_unique_cells
                elif allow_explore and stuck and (frame_idx - last_escape_frame) >= int(args.escape_cooldown_frames):
                    fsm_state = "ESCAPE"
                    fsm_state_frames = 0
                    heading_lock = None
                    turn_commit_sign = None

            elif fsm_state == "TURN_IN_PLACE":
                corner_lock_level += 1
                turn_locked = bool(
                    fsm_state_frames >= int(args.turn_corner_lock_frames) or zero_move_streak >= int(args.zero_move_recovery_streak)
                )
                if fwd > float(args.turn_exit_fwd_m) and fsm_state_frames >= int(args.turn_min_frames):
                    fsm_state = "WALL_FOLLOW"
                    fsm_state_frames = 0
                    corner_lock_level = 0
                    turn_commit_sign = None
                elif (
                    (not bool(startup_wall_guard_active))
                    and (
                        zero_move_streak >= int(args.zero_move_escape_streak)
                        or coverage_stall_streak >= int(args.coverage_stall_escape_streak)
                    )
                    and (frame_idx - last_escape_frame) >= int(args.escape_cooldown_frames)
                ):
                    fsm_state = "ESCAPE"
                    fsm_state_frames = 0
                    heading_lock = None
                    corner_lock_level = 0
                    turn_commit_sign = None
                elif (
                    (not bool(startup_wall_guard_active))
                    and turn_locked
                    and corner_lock_level >= 2
                    and (frame_idx - last_escape_frame) >= int(args.escape_cooldown_frames)
                ):
                    fsm_state = "ESCAPE"
                    fsm_state_frames = 0
                    heading_lock = None
                    corner_lock_level = 0
                    turn_commit_sign = None
                elif (
                    (not bool(startup_wall_guard_active))
                    and stuck
                    and fsm_state_frames >= int(args.turn_to_escape_frames)
                    and (frame_idx - last_escape_frame) >= int(args.escape_cooldown_frames)
                ):
                    fsm_state = "ESCAPE"
                    fsm_state_frames = 0
                    heading_lock = None
                    corner_lock_level = 0
                    turn_commit_sign = None

            elif fsm_state == "ROOM_CHECK_COMMIT":
                if commit_dist_left_m <= 0.0 or commit_frames_left <= 0:
                    if visited_unique_cells > room_check_start_cells:
                        room_check_success_total += 1
                    fsm_state = "WALL_FOLLOW"
                    fsm_state_frames = 0
                    heading_lock = None
                    turn_commit_sign = None

            elif fsm_state == "ESCAPE":
                if heading_lock is None:
                    if escape_recovery_frames_left > 0:
                        scan_phase = "recover"
                    else:
                        scan_phase = "rotate"
                        scan = scan_headings(
                            sim=sim,
                            agent=agent,
                            pos=pos,
                            yaw_ref=yaw,
                            pitch_rad=pitch,
                            visited=visited,
                            forbid_yaw_rad=(float(no_back_yaw_rad) if (no_back_yaw_rad is not None and frame_idx <= int(no_back_until_frame)) else None),
                            forbid_half_angle_rad=math.radians(float(args.no_back_forbid_half_deg)),
                            args=args,
                        )
                        best = scan.get("best")
                        top = scan.get("top") or []
                        escape_attempts_total += 1
                        escape_attempts_consecutive += 1
                        last_escape_frame = frame_idx

                        if best is None:
                            fail_reason = "no_novel_direction"
                            fail_reason_detail = "escape scan returned no heading"
                            done_reason = "FAIL"
                            terminal_record.update(
                                {
                                    "last_fsm_state": fsm_state,
                                    "gate_name": "escape_scan",
                                    "fail_reason": fail_reason,
                                }
                            )
                            break

                        heading_lock = float(best["yaw_rad"])
                        scan_score_best = float(best["score"])
                        scan_best_heading_deg = float(best["yaw_deg"])
                        commit_dist_left_m = float(args.escape_commit_dist_m)
                        commit_frames_left = int(args.escape_commit_frames_max)
                        escape_start_cells = visited_unique_cells
                        if top:
                            # keep only first candidate score for record; full dump is heavy
                            pass
                else:
                    scan_phase = "commit"
                    if commit_dist_left_m <= 0.0 or commit_frames_left <= 0:
                        if visited_unique_cells > (escape_start_cells + int(args.escape_success_new_cells_min)):
                            escape_success_total += 1
                            escape_attempts_consecutive = 0
                        if escape_attempts_consecutive > int(args.max_escape_attempts):
                            fail_reason = "escape_failed"
                            fail_reason_detail = (
                                f"escape_attempts_consecutive={escape_attempts_consecutive} > max_escape_attempts={int(args.max_escape_attempts)}"
                            )
                            done_reason = "FAIL"
                            terminal_record.update(
                                {
                                    "last_fsm_state": fsm_state,
                                    "gate_name": "escape_attempt_limit",
                                    "fail_reason": fail_reason,
                                }
                            )
                            break
                        fsm_state = "WALL_FOLLOW"
                        fsm_state_frames = 0
                        heading_lock = None
                        turn_commit_sign = None

            if fsm_state == "ESCAPE" and fsm_prev_state != "ESCAPE":
                no_back_yaw_rad = float(last_effective_move_yaw)
                no_back_until_frame = frame_idx + int(args.no_back_forbid_frames)

            recovery_used = "none"
            motion_policy = "wall_follow"
            motion_source = "direct_pose"
            nav_step_error = None
            yaw_cmd = 0.0
            step_cmd = 0.0
            move_mode = "forward"

            if fsm_state == "TURN_IN_PLACE":
                motion_policy = "turn_in_place"
                if turn_commit_sign is None:
                    turn_commit_sign = choose_turn_commit_sign(left=left, right=right, delta_l=delta_l, delta_f=delta_f)
                yaw_cmd = float(turn_commit_sign) * math.radians(float(args.turn_yaw_limit_deg))
                step_cmd = 0.0
            elif fsm_state == "ROOM_CHECK_COMMIT":
                motion_policy = "room_check_commit"
                target_yaw = yaw if heading_lock is None else float(heading_lock)
                yaw_cmd = clamp(
                    angle_diff_rad(target_yaw, yaw),
                    -math.radians(float(args.commit_yaw_limit_deg)),
                    math.radians(float(args.commit_yaw_limit_deg)),
                )
                step_cmd = compute_step_length(fwd=fwd, state=fsm_state, args=args)
            elif fsm_state == "ESCAPE":
                if heading_lock is None and escape_recovery_frames_left > 0:
                    motion_policy = "escape_recover"
                    yaw_cmd = 0.0
                    if escape_recovery_mode == "backoff":
                        recovery_used = "backoff"
                        move_mode = "backward"
                        backoff_mult = (
                            int(args.black_backoff_steps_heavy)
                            if (black_soft_alert or outside_proxy or black_alert)
                            else int(args.black_backoff_steps_medium)
                        )
                        step_cmd = min(
                            float(args.black_backoff_step_m) * max(1, backoff_mult),
                            float(args.recovery_step_cap_m),
                        )
                    else:
                        recovery_used = "sidestep"
                        move_mode = "sidestep_left" if left >= right else "sidestep_right"
                        step_cmd = max(float(args.step_min_m), float(args.corner_sidestep_m))
                else:
                    motion_policy = "escape"
                    target_yaw = yaw if heading_lock is None else float(heading_lock)
                    yaw_cmd = clamp(
                        angle_diff_rad(target_yaw, yaw),
                        -math.radians(float(args.escape_commit_yaw_limit_deg)),
                        math.radians(float(args.escape_commit_yaw_limit_deg)),
                    )
                    step_cmd = compute_step_length(fwd=fwd, state=fsm_state, args=args)
                    if heading_lock is not None and step_cmd <= 0.0:
                        if max(left, right) > float(args.step_obstacle_margin_m) + 0.08:
                            recovery_used = "sidestep"
                            move_mode = "sidestep_left" if left >= right else "sidestep_right"
                            step_cmd = max(float(args.step_min_m), float(args.corner_sidestep_m))
                        else:
                            recovery_used = "backoff"
                            move_mode = "backward"
                            step_cmd = min(float(args.inside_backoff_m), float(args.recovery_step_cap_m))
            else:
                motion_policy = "wall_follow"
                e = float(left - float(args.wall_target_dist_m))
                yaw_cmd = clamp(
                    -float(args.wall_kp) * e,
                    -math.radians(float(args.wall_yaw_limit_deg)),
                    math.radians(float(args.wall_yaw_limit_deg)),
                )
                if geom_event == "CORNER_INNER" or fwd < float(args.turn_enter_fwd_m):
                    step_cmd = 0.0
                else:
                    step_cmd = compute_step_length(fwd=fwd, state=fsm_state, args=args)
                    if zero_move_streak >= int(args.zero_move_recovery_streak):
                        recovery_used = "sidestep"
                        move_mode = "sidestep_right" if left <= right else "sidestep_left"
                        step_cmd = max(step_cmd, float(args.corner_sidestep_m))

            yaw_next = wrap_angle_rad(yaw + yaw_cmd)
            if move_mode == "forward":
                step_cap = max(0.0, float(fwd) - float(args.step_obstacle_margin_m))
            elif move_mode == "backward":
                step_cap = float(args.recovery_step_cap_m)
            elif move_mode == "sidestep_left":
                step_cap = max(0.0, float(left) - float(args.step_obstacle_margin_m))
            else:
                step_cap = max(0.0, float(right) - float(args.step_obstacle_margin_m))
            step_final = min(step_cmd, step_cap)
            if step_cmd > 0.0 and step_final <= 0.0:
                recovery_used = "rotate_probe_big"

            pos_target = pos.copy()
            if step_final > 0.0:
                dir_x = math.sin(yaw_next)
                dir_z = -math.cos(yaw_next)
                if move_mode == "forward":
                    pos_target[0] += dir_x * step_final
                    pos_target[2] += dir_z * step_final
                elif move_mode == "backward":
                    pos_target[0] -= dir_x * step_final
                    pos_target[2] -= dir_z * step_final
                elif move_mode == "sidestep_left":
                    pos_target[0] += -math.cos(yaw_next) * step_final
                    pos_target[2] += -math.sin(yaw_next) * step_final
                else:
                    pos_target[0] += math.cos(yaw_next) * step_final
                    pos_target[2] += math.sin(yaw_next) * step_final
            if inside_force_reanchor:
                recovery_used = "reanchor"
                move_mode = "reanchor"
                pos_target = origin_pos.copy()
                yaw_next = float(origin_yaw)
                step_final = 0.0
            elif bool(use_navmesh_move) and step_final > 0.0:
                nav_target, nav_step_error = try_navmesh_step(
                    pathfinder=sim.pathfinder,
                    start=pos.astype(np.float32),
                    target=pos_target.astype(np.float32),
                )
                if nav_target is not None:
                    pos_target = nav_target
                    motion_source = "navmesh_try_step"
                else:
                    motion_source = "direct_pose_fallback"

            ts_policy = time.time()
            obs_after = observe_pose(sim=sim, agent=agent, position=pos_target.astype(np.float32), yaw_rad=yaw_next, pitch_rad=pitch)
            timings_policy = float(time.time() - ts_policy)

            pos_after = np.asarray(obs_after["pose_readback"]["position"], dtype=np.float64)
            yaw_after = float(yaw_next)

            delta_pos_m = float(np.linalg.norm(pos_after - pos))
            delta_yaw_deg = float(abs(math.degrees(angle_diff_rad(yaw_after, yaw))))
            delta_h_m = float(abs((pos_after[1] - floor_height) - (pos[1] - floor_height)))

            if step_final > float(args.blocked_step_min_m) and delta_pos_m < float(args.blocked_delta_pos_m):
                recovery_used = "rotate_probe_big"

            path_length_m += delta_pos_m
            if delta_pos_m >= float(args.last_move_min_delta_m):
                last_effective_move_yaw = float(yaw_after)
            if delta_pos_m < float(args.zero_move_delta_pos_m):
                zero_move_streak += 1
            else:
                zero_move_streak = 0
            max_zero_move_streak = max(max_zero_move_streak, zero_move_streak)
            if fsm_state in {"ROOM_CHECK_COMMIT", "ESCAPE"}:
                commit_dist_left_m = max(0.0, commit_dist_left_m - delta_pos_m)
                commit_frames_left -= 1
            if fsm_state != "TURN_IN_PLACE":
                corner_lock_level = 0
                turn_commit_sign = None

            if fsm_state == "ESCAPE" and heading_lock is not None:
                if delta_pos_m < float(args.zero_move_delta_pos_m):
                    escape_blocked_streak += 1
                else:
                    escape_blocked_streak = 0
                max_escape_blocked_streak = max(max_escape_blocked_streak, int(escape_blocked_streak))
                if escape_blocked_streak >= int(args.escape_blocked_rescan_streak):
                    no_back_yaw_rad = float(heading_lock)
                    no_back_until_frame = frame_idx + int(args.no_back_forbid_frames)
                    heading_lock = None
                    commit_dist_left_m = 0.0
                    commit_frames_left = 0
                    escape_blocked_streak = 0
            else:
                escape_blocked_streak = 0

            if fsm_state == "ESCAPE" and heading_lock is None and escape_recovery_frames_left > 0:
                escape_recovery_frames_left = max(0, int(escape_recovery_frames_left) - 1)
                if escape_recovery_mode == "backoff" and escape_recovery_frames_left > 0:
                    escape_recovery_mode = "sidestep"
                elif escape_recovery_frames_left <= 0:
                    escape_recovery_mode = "none"

            plateau_window.append(
                {
                    "path": float(path_length_m),
                    "visited": int(visited_unique_cells),
                    "pos": pos_after.copy(),
                }
            )
            while len(plateau_window) >= 2 and (path_length_m - float(plateau_window[0]["path"])) > float(args.plateau_window_path_m):
                plateau_window.popleft()
            path_span_last_window = 0.0
            coverage_gain_last_path_m = 0
            net_disp_last_path_m = 0.0
            coverage_plateau = False
            coverage_done = False
            if plateau_window:
                base = plateau_window[0]
                path_span_last_window = float(path_length_m - float(base["path"]))
                coverage_gain_last_path_m = int(visited_unique_cells - int(base["visited"]))
                net_disp_last_path_m = float(np.linalg.norm(pos_after - np.asarray(base["pos"], dtype=np.float64)))
                coverage_plateau = bool(
                    path_span_last_window >= float(args.plateau_window_path_m) * 0.8
                    and coverage_gain_last_path_m <= int(args.plateau_new_cells_max)
                    and net_disp_last_path_m >= float(args.plateau_min_net_disp_m)
                    and escape_attempts_consecutive <= int(args.plateau_max_escape_attempts_consecutive)
                )
                coverage_done = bool(
                    path_length_m >= float(args.path_min_done_m)
                    and (
                        coverage_plateau
                        or (
                            coverage_area_m2 >= float(args.path_coverage_done_min_m)
                            and coverage_gain_last_path_m <= int(args.plateau_new_cells_max)
                        )
                    )
                )

            state_pos_limit = float(args.wall_delta_pos_max_m)
            state_yaw_limit = float(args.wall_delta_yaw_max_deg)
            if fsm_state == "TURN_IN_PLACE":
                state_pos_limit = float(args.turn_delta_pos_max_m)
                state_yaw_limit = float(args.turn_delta_yaw_max_deg)
            elif fsm_state in {"ROOM_CHECK_COMMIT", "ESCAPE"}:
                state_pos_limit = float(args.commit_delta_pos_max_m)
                state_yaw_limit = float(args.commit_delta_yaw_max_deg)
            if recovery_used == "reanchor":
                state_pos_limit = max(state_pos_limit, float(args.reanchor_delta_pos_max_m))
                state_yaw_limit = max(state_yaw_limit, float(args.reanchor_delta_yaw_max_deg))

            continuity_status = "OK"
            continuity_fail_reason = None
            continuity_fail_detail = None
            gate_fail = False
            gate_fail_items = []
            if delta_h_m > float(args.delta_h_max_m):
                gate_fail = True
                gate_fail_items.append(f"delta_h_m={delta_h_m:.4f}>{float(args.delta_h_max_m):.4f}")
            if delta_pos_m > state_pos_limit:
                gate_fail = True
                gate_fail_items.append(f"delta_pos_m={delta_pos_m:.4f}>{state_pos_limit:.4f}")
            if delta_yaw_deg > state_yaw_limit:
                gate_fail = True
                gate_fail_items.append(f"delta_yaw_deg={delta_yaw_deg:.3f}>{state_yaw_limit:.3f}")

            if gate_fail:
                continuity_status = "RECOVERY_OK"
                continuity_fail_reason = "continuity_gate_failed"
                continuity_fail_detail = "; ".join(gate_fail_items)
                fail_reason_hist[continuity_fail_reason] += 1
                consecutive_recovery += 1
                recovery_used = recovery_used if recovery_used != "none" else "sidestep"
                if (frame_idx - last_escape_frame) >= int(args.escape_cooldown_frames):
                    fsm_state = "ESCAPE"
                    fsm_state_frames = 0
                    heading_lock = None
                if consecutive_recovery > int(args.max_consecutive_recovery_fail):
                    fail_reason = "continuity_gate_failed"
                    fail_reason_detail = continuity_fail_detail
                    done_reason = "FAIL"
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "continuity_gate",
                            "fail_reason": fail_reason,
                        }
                    )
                    append_jsonl(
                        poses_path,
                        {
                            "frame_idx": frame_idx,
                            "terminal_record": True,
                            "last_fsm_state": fsm_state,
                            "continuity_fail_reason": continuity_fail_reason,
                            "continuity_fail_detail": continuity_fail_detail,
                        },
                    )
                    break
            else:
                continuity_ok_frames += 1
                if recovery_used == "none":
                    consecutive_recovery = 0

            if recovery_used != "none":
                recovery_frames += 1
            max_consecutive_recovery = max(max_consecutive_recovery, consecutive_recovery)

            if min_depth > 0 and min_depth < float(args.collision_min_depth_m):
                collision_frames += 1

            frame_elapsed = float(time.time() - frame_t0)
            frame_record = {
                "frame_idx": frame_idx,
                "continuity_status": continuity_status,
                "continuity_fail_reason": continuity_fail_reason,
                "continuity_fail_detail": continuity_fail_detail,
                "fsm_state": fsm_state,
                "fsm_prev_state": fsm_prev_state,
                "fsm_state_frames": int(fsm_state_frames),
                "motion_policy": motion_policy,
                "geom_event": geom_event,
                "delta_l": delta_l,
                "delta_f": delta_f,
                "delta_pos_m": delta_pos_m,
                "delta_yaw_deg": delta_yaw_deg,
                "delta_h_m": delta_h_m,
                "black_ratio": black_ratio,
                "black_void_ratio": float(black_void_ratio),
                "black_soft_alert": bool(black_soft_alert),
                "black_soft_streak": int(black_soft_streak),
                "depth_valid_ratio": valid_ratio,
                "outside_proxy": bool(outside_proxy),
                "wall_lock": bool(wall_lock),
                "opening_persist_count": int(opening_persist_count),
                "startup_wall_guard_active": bool(startup_wall_guard_active),
                "no_back_active": bool(no_back_yaw_rad is not None and frame_idx <= int(no_back_until_frame)),
                "no_back_yaw_deg": None if no_back_yaw_rad is None else float(math.degrees(float(no_back_yaw_rad))),
                "zero_move_streak": int(zero_move_streak),
                "coverage_stall_streak": int(coverage_stall_streak),
                "clearance_fwd_m": fwd,
                "clearance_left_m": left,
                "clearance_right_m": right,
                "clearance_fwd_left_m": fwd_left,
                "min_depth_m": min_depth,
                "near_wall_ratio": near_wall_ratio,
                "novelty_score": novelty_score,
                "frontier_hits_norm": frontier_hits_norm,
                "room_check_trigger": bool(room_check_trigger),
                "room_check_heading_deg": None if heading_lock is None else float(math.degrees(heading_lock)),
                "scan_score_best": scan_score_best,
                "scan_best_heading_deg": scan_best_heading_deg,
                "scan_phase": scan_phase,
                "commit_dist_left_m": float(commit_dist_left_m),
                "commit_frames_left": int(commit_frames_left),
                "recovery_used": recovery_used,
                "motion_source": motion_source,
                "nav_step_error": nav_step_error,
                "turn_commit_sign": None if turn_commit_sign is None else float(turn_commit_sign),
                "escape_recovery_mode": escape_recovery_mode,
                "turn_locked": bool(turn_locked),
                "collision_proxy": {
                    "near_wall_ratio": near_wall_ratio,
                    "min_depth_m": min_depth,
                    "forward_clearance_m": fwd,
                },
                "pose": {
                    "position": pos_after.tolist(),
                    "yaw_rad": yaw_after,
                    "pitch_rad": pitch,
                },
                "path_length_m": path_length_m,
                "visited_unique_cells": int(visited_unique_cells),
                "coverage_area_m2": coverage_area_m2,
                "new_cell": int(new_cell),
                "coverage_gain_last_window": int(coverage_gain_last_path_m),
                "net_disp_last_window_m": float(net_disp_last_path_m),
                "coverage_plateau": bool(coverage_plateau),
                "coverage_done": bool(coverage_done),
                "progress_marker": "frame_done",
                "stage_timing_sec": {
                    "t_step3_render": timings_render,
                    "t_step3_policy": timings_policy,
                    "t_step3_total": frame_elapsed,
                },
            }
            append_jsonl(poses_path, frame_record)
            traj_records += 1

            pos = pos_after
            yaw = yaw_after

            if fsm_state == fsm_prev_state:
                fsm_state_frames += 1
            else:
                fsm_state_frames = 1

            # optional loop closure finish
            if bool(args.enable_origin_finish):
                origin_dist = float(np.linalg.norm(pos - origin_pos))
                origin_yaw_diff = abs(math.degrees(angle_diff_rad(yaw, origin_yaw)))
                if (
                    origin_dist < float(args.origin_finish_pos_m)
                    and origin_yaw_diff < float(args.origin_finish_yaw_deg)
                    and path_length_m >= float(args.path_min_done_m)
                    and coverage_plateau
                ):
                    origin_match_streak += 1
                else:
                    origin_match_streak = 0
                if origin_match_streak >= int(args.origin_finish_streak):
                    done_reason = "DONE"
                    plateau_reason = "origin_finish"
                    break

            # regular finish
            if path_length_m >= float(args.path_max_m):
                done_reason = "DONE"
                plateau_reason = "path_max_reached"
                break
            if coverage_done:
                done_reason = "DONE"
                plateau_reason = "coverage_done"
                break
            if (frame_idx + 1) >= int(args.target_frames):
                if coverage_done:
                    done_reason = "DONE"
                    plateau_reason = "frame_budget_coverage_done"
                elif path_length_m < float(args.path_min_done_m):
                    fail_reason = "stuck_no_progress"
                    fail_reason_detail = (
                        f"path_length_m={path_length_m:.3f}<path_min_done_m={float(args.path_min_done_m):.3f} "
                        f"at_target_frames={int(args.target_frames)}"
                    )
                    done_reason = "FAIL"
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "frame_budget_min_path_gate",
                            "fail_reason": fail_reason,
                        }
                    )
                else:
                    done_reason = "TIMEOUT"
                    fail_reason = "target_frames_reached"
                    fail_reason_detail = (
                        f"path_length_m={path_length_m:.3f}, path_min_m={float(args.path_min_done_m):.3f}, "
                        f"coverage_gain_last_window={int(coverage_gain_last_path_m)}, net_disp_last_window_m={net_disp_last_path_m:.3f}"
                    )
                    terminal_record.update(
                        {
                            "last_fsm_state": fsm_state,
                            "gate_name": "frame_budget",
                            "fail_reason": fail_reason,
                        }
                    )
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
                fail_reason_detail = f"target_frames={int(args.target_frames)} reached before done gate"
        elif done_reason == "FAIL":
            status = "FAIL"
            run_state = "FAIL"
            step3_ok = False
            if fail_reason is None:
                fail_reason = "stuck_no_progress"
        else:
            status = "FAIL"
            run_state = "FAIL"
            step3_ok = False
            if fail_reason is None:
                fail_reason = "stuck_no_progress"
                fail_reason_detail = "loop_ended_without_done"

    except Exception as exc:
        status = "FAIL"
        run_state = "FAIL"
        step3_ok = False
        fail_reason = "STEP3_EXCEPTION"
        fail_reason_detail = str(exc)
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass

    frames = int(traj_records)
    visited_unique_cells = int(len(visited))
    coverage_area_m2 = float(visited_unique_cells) * float(args.grid_res_m) * float(args.grid_res_m)
    continuity_valid_ratio = float(continuity_ok_frames / max(frames, 1))
    recovery_ratio = float(recovery_frames / max(frames, 1))
    collision_ratio = float(collision_frames / max(frames, 1))
    avg_step_m = float(path_length_m / max(frames, 1))
    net_displacement_m = float(np.linalg.norm(pos - origin_pos)) if frames > 0 else 0.0
    black_void_ratio_avg = float(sum_black_void_ratio / max(frames, 1))
    wall_lock_ratio = float(wall_lock_frames / max(frames, 1))
    startup_wall_guard_ratio = float(startup_wall_guard_frames / max(frames, 1))

    room_check_success_ratio = float(room_check_success_total / max(room_check_triggers_total, 1))
    escape_success_ratio = float(escape_success_total / max(escape_attempts_total, 1))

    if not step3_ok and fail_reason:
        fail_reason_hist[fail_reason] += 1

    traj_meta = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 3,
        "step_name": "step3_explorer_fsm",
        "step2_report": str(step2_report_path),
        "status": status,
        "run_state": run_state,
        "step3_ok": bool(step3_ok),
        "navmesh_probe": navmesh_probe,
        "use_navmesh_move": bool(use_navmesh_move),
        "frames": frames,
        "continuity_valid_ratio": continuity_valid_ratio,
        "recovery_ratio": recovery_ratio,
        "collision_ratio": collision_ratio,
        "max_consecutive_recovery": int(max_consecutive_recovery),
        "max_zero_move_streak": int(max_zero_move_streak),
        "max_coverage_stall_streak": int(max_coverage_stall_streak),
        "max_escape_blocked_streak": int(max_escape_blocked_streak),
        "black_alert_frames": int(black_alert_frames),
        "black_soft_alert_frames": int(black_soft_alert_frames),
        "outside_proxy_frames": int(outside_proxy_frames),
        "inside_reanchor_count": int(inside_reanchor_count),
        "black_void_ratio": black_void_ratio_avg,
        "black_soft_alert": bool(black_soft_alert_frames > 0),
        "black_soft_streak": int(max_black_soft_streak),
        "wall_lock": bool(last_wall_lock),
        "wall_lock_ratio": wall_lock_ratio,
        "opening_persist_count": int(max_opening_persist_count),
        "startup_wall_guard_active": bool(last_startup_wall_guard_active),
        "startup_wall_guard_ratio": startup_wall_guard_ratio,
        "path_length_m": float(path_length_m),
        "avg_step_m": avg_step_m,
        "net_displacement_m": net_displacement_m,
        "visited_unique_cells": visited_unique_cells,
        "coverage_area_m2": coverage_area_m2,
        "coverage_gain_recent": int(coverage_gain_recent),
        "coverage_gain_last_window": int(coverage_gain_last_path_m),
        "net_disp_last_window_m": float(net_disp_last_path_m),
        "plateau_reason": plateau_reason,
        "room_check_triggers_total": int(room_check_triggers_total),
        "room_check_success_ratio": room_check_success_ratio,
        "escape_attempts_total": int(escape_attempts_total),
        "escape_success_ratio": escape_success_ratio,
        "escape_attempts_consecutive": int(escape_attempts_consecutive),
        "fail_reason_histogram": dict(fail_reason_hist),
        "fail_reason": fail_reason,
        "fail_reason_detail": fail_reason_detail,
        "terminal_record": {
            "last_fsm_state": fsm_state,
            "gate_name": terminal_record.get("gate_name"),
            "fail_reason": terminal_record.get("fail_reason") or fail_reason,
            "counters_snapshot": {
                "room_check_triggers_total": int(room_check_triggers_total),
                "escape_attempts_total": int(escape_attempts_total),
                "max_consecutive_recovery": int(max_consecutive_recovery),
                "coverage_gain_recent": int(coverage_gain_recent),
                "coverage_gain_last_window": int(coverage_gain_last_path_m),
                "net_disp_last_window_m": float(net_disp_last_path_m),
                "max_zero_move_streak": int(max_zero_move_streak),
                "max_coverage_stall_streak": int(max_coverage_stall_streak),
                "black_alert_frames": int(black_alert_frames),
                "outside_proxy_frames": int(outside_proxy_frames),
                "inside_reanchor_count": int(inside_reanchor_count),
            },
            "stuck_window": {
                "net_disp_window": float(net_disp_window) if 'net_disp_window' in locals() else None,
                "path_window": float(path_window) if 'path_window' in locals() else None,
                "new_cells_window": int(new_cells_window) if 'new_cells_window' in locals() else None,
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
        "progress_marker": "done" if step3_ok else "fail",
        "timeout_stage": None,
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
            "navmesh_probe": navmesh_probe,
            "use_navmesh_move": bool(use_navmesh_move),
            "frames": frames,
            "path_length_m": float(path_length_m),
            "visited_unique_cells": visited_unique_cells,
            "coverage_area_m2": coverage_area_m2,
            "continuity_valid_ratio": continuity_valid_ratio,
            "recovery_ratio": recovery_ratio,
            "max_zero_move_streak": int(max_zero_move_streak),
            "max_coverage_stall_streak": int(max_coverage_stall_streak),
            "black_alert_frames": int(black_alert_frames),
            "outside_proxy_frames": int(outside_proxy_frames),
            "inside_reanchor_count": int(inside_reanchor_count),
            "room_check_triggers_total": int(room_check_triggers_total),
            "escape_attempts_total": int(escape_attempts_total),
            "fail_reason": fail_reason,
            "fail_reason_detail": fail_reason_detail,
            "coverage_gain_last_window": int(coverage_gain_last_path_m),
            "net_disp_last_window_m": float(net_disp_last_path_m),
            "plateau_reason": plateau_reason,
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
        "frames": frames,
        "path_length_m": float(path_length_m),
        "visited_unique_cells": visited_unique_cells,
        "coverage_area_m2": coverage_area_m2,
        "continuity_valid_ratio": continuity_valid_ratio,
        "recovery_ratio": recovery_ratio,
        "room_check_triggers_total": int(room_check_triggers_total),
        "escape_attempts_total": int(escape_attempts_total),
        "fail_reason": fail_reason,
        "traj_meta_json": str(traj_meta_path),
        "step3_report_json": str(report_path),
    }


def parse_step3_report_summary(report_path: Path) -> Dict:
    payload = load_json(report_path)
    if not isinstance(payload, dict):
        return {
            "status": "FAIL",
            "run_state": "MISSING",
            "step3_ok": False,
            "frames": 0,
            "path_length_m": 0.0,
            "visited_unique_cells": 0,
            "coverage_area_m2": 0.0,
            "continuity_valid_ratio": 0.0,
            "recovery_ratio": 0.0,
            "room_check_triggers_total": 0,
            "escape_attempts_total": 0,
            "fail_reason": "step3_report_missing_or_invalid",
            "traj_meta_json": "",
        }
    return {
        "status": str(payload.get("status") or "FAIL"),
        "run_state": str(payload.get("run_state") or "FAIL"),
        "step3_ok": bool(payload.get("step3_ok", False)),
        "frames": int(payload.get("frames") or 0),
        "path_length_m": float(payload.get("path_length_m") or 0.0),
        "visited_unique_cells": int(payload.get("visited_unique_cells") or 0),
        "coverage_area_m2": float(payload.get("coverage_area_m2") or 0.0),
        "continuity_valid_ratio": float(payload.get("continuity_valid_ratio") or 0.0),
        "recovery_ratio": float(payload.get("recovery_ratio") or 0.0),
        "room_check_triggers_total": int(payload.get("room_check_triggers_total") or 0),
        "escape_attempts_total": int(payload.get("escape_attempts_total") or 0),
        "fail_reason": payload.get("fail_reason"),
        "traj_meta_json": str(payload.get("traj_meta_json") or ""),
    }


def synthesize_step3_report_for_crash(
    report_path: Path,
    scene_id: str,
    scene_path: str,
    step2_report: Path,
    run_state: str,
    fail_reason: str,
    worker_exit_code: Optional[int],
    log_path: Path,
    env_meta: Dict,
):
    payload = {
        "scene_id": scene_id,
        "scene_path": scene_path,
        "pipeline_stage": 3,
        "step_name": "step3_explorer_fsm",
        "status": run_state,
        "run_state": run_state,
        "step3_ok": False,
        "fail_reason": fail_reason,
        "step2_report": str(step2_report),
        "worker_exit_code": worker_exit_code,
        "log_path": str(log_path),
        "environment": env_meta,
    }
    save_json(report_path, payload)


def discover_step2_reports(step2_root: Path, scene_id_filter: Optional[str]) -> List[Path]:
    if not step2_root.exists():
        return []
    rows = []
    for p in sorted(step2_root.glob("*/step2_floor_report.json")):
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
        "--grid-res-m",
        str(args.grid_res_m),
        "--novelty-cap-m",
        str(args.novelty_cap_m),
        "--novelty-sample-step-m",
        str(args.novelty_sample_step_m),
        "--lambda-novelty",
        str(args.lambda_novelty),
        "--lambda-frontier",
        str(args.lambda_frontier),
        "--wall-target-dist-m",
        str(args.wall_target_dist_m),
        "--wall-kp",
        str(args.wall_kp),
        "--wall-yaw-limit-deg",
        str(args.wall_yaw_limit_deg),
        "--turn-yaw-limit-deg",
        str(args.turn_yaw_limit_deg),
        "--commit-yaw-limit-deg",
        str(args.commit_yaw_limit_deg),
        "--escape-commit-yaw-limit-deg",
        str(args.escape_commit_yaw_limit_deg),
        "--room-check-commit-dist-m",
        str(args.room_check_commit_dist_m),
        "--escape-commit-dist-m",
        str(args.escape_commit_dist_m),
        "--commit-frames-max",
        str(args.commit_frames_max),
        "--escape-commit-frames-max",
        str(args.escape_commit_frames_max),
        "--escape-scan-bins",
        str(args.escape_scan_bins),
        "--escape-scan-top-k",
        str(args.escape_scan_top_k),
        "--max-escape-attempts",
        str(args.max_escape_attempts),
        "--escape-cooldown-frames",
        str(args.escape_cooldown_frames),
        "--stuck-window-frames",
        str(args.stuck_window_frames),
        "--stuck-net-disp-m",
        str(args.stuck_net_disp_m),
        "--stuck-path-m",
        str(args.stuck_path_m),
        "--stuck-new-cells-max",
        str(args.stuck_new_cells_max),
        "--stuck-near-wall-ratio",
        str(args.stuck_near_wall_ratio),
        "--zero-move-delta-pos-m",
        str(args.zero_move_delta_pos_m),
        "--zero-move-recovery-streak",
        str(args.zero_move_recovery_streak),
        "--zero-move-escape-streak",
        str(args.zero_move_escape_streak),
        "--coverage-stall-new-cells-max",
        str(args.coverage_stall_new_cells_max),
        "--coverage-stall-escape-streak",
        str(args.coverage_stall_escape_streak),
        "--corner-strong-f-m",
        str(args.corner_strong_f_m),
        "--corner-strong-fl-m",
        str(args.corner_strong_fl_m),
        "--corner-delta-f-m",
        str(args.corner_delta_f_m),
        "--corner-delta-fl-m",
        str(args.corner_delta_fl_m),
        "--corner-near-f-m",
        str(args.corner_near_f_m),
        "--opening-delta-l-m",
        str(args.opening_delta_l_m),
        "--opening-left-m",
        str(args.opening_left_m),
        "--opening-fwd-m",
        str(args.opening_fwd_m),
        "--opening-votes-needed",
        str(args.opening_votes_needed),
        "--opening-right-cap-m",
        str(args.opening_right_cap_m),
        "--opening-min-valid-ratio",
        str(args.opening_min_valid_ratio),
        "--opening-override-delta-l-m",
        str(args.opening_override_delta_l_m),
        "--opening-override-fwd-m",
        str(args.opening_override_fwd_m),
        "--opening-persist-frames",
        str(args.opening_persist_frames),
        "--step-nominal-m",
        str(args.step_nominal_m),
        "--step-nominal-wall-m",
        str(args.step_nominal_wall_m),
        "--step-nominal-commit-m",
        str(args.step_nominal_commit_m),
        "--step-nominal-escape-m",
        str(args.step_nominal_escape_m),
        "--step-min-m",
        str(args.step_min_m),
        "--step-max-m",
        str(args.step_max_m),
        "--step-clearance-scale",
        str(args.step_clearance_scale),
        "--step-clearance-scale-wall",
        str(args.step_clearance_scale_wall),
        "--step-clearance-scale-commit",
        str(args.step_clearance_scale_commit),
        "--step-clearance-scale-escape",
        str(args.step_clearance_scale_escape),
        "--step-obstacle-margin-m",
        str(args.step_obstacle_margin_m),
        "--anti-freeze-fwd-m",
        str(args.anti_freeze_fwd_m),
        "--anti-freeze-step-m",
        str(args.anti_freeze_step_m),
        "--step-block-fwd-m",
        str(args.step_block_fwd_m),
        "--last-move-min-delta-m",
        str(args.last_move_min_delta_m),
        "--black-backoff-step-m",
        str(args.black_backoff_step_m),
        "--black-backoff-steps-medium",
        str(args.black_backoff_steps_medium),
        "--black-backoff-steps-heavy",
        str(args.black_backoff_steps_heavy),
        "--black-recovery-commit-dist-m",
        str(args.black_recovery_commit_dist_m),
        "--black-recovery-commit-min-frames",
        str(args.black_recovery_commit_min_frames),
        "--black-lateral-primary-deg",
        str(args.black_lateral_primary_deg),
        "--black-lateral-secondary-deg",
        str(args.black_lateral_secondary_deg),
        "--black-lateral-secondary-when-tight-m",
        str(args.black_lateral_secondary_when_tight_m),
        "--no-back-forbid-half-deg",
        str(args.no_back_forbid_half_deg),
        "--no-back-forbid-frames",
        str(args.no_back_forbid_frames),
        "--no-back-forbid-score-scale",
        str(args.no_back_forbid_score_scale),
        "--escape-blocked-rescan-streak",
        str(args.escape_blocked_rescan_streak),
        "--turn-enter-fwd-m",
        str(args.turn_enter_fwd_m),
        "--turn-exit-fwd-m",
        str(args.turn_exit_fwd_m),
        "--turn-min-frames",
        str(args.turn_min_frames),
        "--turn-to-escape-frames",
        str(args.turn_to_escape_frames),
        "--startup-wall-frames",
        str(args.startup_wall_frames),
        "--startup-wall-path-m",
        str(args.startup_wall_path_m),
        "--delta-h-max-m",
        str(args.delta_h_max_m),
        "--wall-delta-pos-max-m",
        str(args.wall_delta_pos_max_m),
        "--commit-delta-pos-max-m",
        str(args.commit_delta_pos_max_m),
        "--turn-delta-pos-max-m",
        str(args.turn_delta_pos_max_m),
        "--wall-delta-yaw-max-deg",
        str(args.wall_delta_yaw_max_deg),
        "--commit-delta-yaw-max-deg",
        str(args.commit_delta_yaw_max_deg),
        "--turn-delta-yaw-max-deg",
        str(args.turn_delta_yaw_max_deg),
        "--max-consecutive-recovery-fail",
        str(args.max_consecutive_recovery_fail),
        "--depth-valid-min-m",
        str(args.depth_valid_min_m),
        "--depth-valid-max-m",
        str(args.depth_valid_max_m),
        "--near-wall-depth-m",
        str(args.near_wall_depth_m),
        "--near-wall-ratio-high",
        str(args.near_wall_ratio_high),
        "--blank-valid-ratio",
        str(args.blank_valid_ratio),
        "--blank-frame-streak-max",
        str(args.blank_frame_streak_max),
        "--black-pixel-threshold",
        str(args.black_pixel_threshold),
        "--black-ratio-recovery",
        str(args.black_ratio_recovery),
        "--black-soft-void-ratio",
        str(args.black_soft_void_ratio),
        "--black-soft-ratio",
        str(args.black_soft_ratio),
        "--black-ratio-fail",
        str(args.black_ratio_fail),
        "--black-fail-streak",
        str(args.black_fail_streak),
        "--depth-valid-ratio-recovery-min",
        str(args.depth_valid_ratio_recovery_min),
        "--depth-valid-ratio-recovery-streak",
        str(args.depth_valid_ratio_recovery_streak),
        "--depth-valid-ratio-fail-streak",
        str(args.depth_valid_ratio_fail_streak),
        "--outside-valid-ratio-min",
        str(args.outside_valid_ratio_min),
        "--outside-center-valid-ratio-max",
        str(args.outside_center_valid_ratio_max),
        "--outside-side-valid-ratio-max",
        str(args.outside_side_valid_ratio_max),
        "--outside-proxy-recovery-streak",
        str(args.outside_proxy_recovery_streak),
        "--outside-proxy-reanchor-streak",
        str(args.outside_proxy_reanchor_streak),
        "--outside-proxy-fail-streak",
        str(args.outside_proxy_fail_streak),
        "--inside-backoff-m",
        str(args.inside_backoff_m),
        "--inside-reanchor-max",
        str(args.inside_reanchor_max),
        "--safety-min-depth-m",
        str(args.safety_min_depth_m),
        "--collision-min-depth-m",
        str(args.collision_min_depth_m),
        "--blocked-step-min-m",
        str(args.blocked_step_min_m),
        "--blocked-delta-pos-m",
        str(args.blocked_delta_pos_m),
        "--recovery-step-cap-m",
        str(args.recovery_step_cap_m),
        "--escape-success-new-cells-min",
        str(args.escape_success_new_cells_min),
        "--plateau-window-path-m",
        str(args.plateau_window_path_m),
        "--plateau-new-cells-max",
        str(args.plateau_new_cells_max),
        "--plateau-min-net-disp-m",
        str(args.plateau_min_net_disp_m),
        "--plateau-max-escape-attempts-consecutive",
        str(args.plateau_max_escape_attempts_consecutive),
        "--coverage-gain-recent-max",
        str(args.coverage_gain_recent_max),
        "--path-min-done-m",
        str(args.path_min_done_m),
        "--hazard-soft-timeout-path-m",
        str(args.hazard_soft_timeout_path_m),
        "--path-coverage-done-min-m",
        str(args.path_coverage_done_min_m),
        "--reanchor-delta-pos-max-m",
        str(args.reanchor_delta_pos_max_m),
        "--reanchor-delta-yaw-max-deg",
        str(args.reanchor_delta_yaw_max_deg),
        "--turn-corner-lock-frames",
        str(args.turn_corner_lock_frames),
        "--corner-backoff-m",
        str(args.corner_backoff_m),
        "--corner-sidestep-m",
        str(args.corner_sidestep_m),
        "--step3-pitch-deg",
        str(args.step3_pitch_deg),
        "--origin-finish-pos-m",
        str(args.origin_finish_pos_m),
        "--origin-finish-yaw-deg",
        str(args.origin_finish_yaw_deg),
        "--origin-finish-streak",
        str(args.origin_finish_streak),
        "--video-fps",
        str(args.video_fps),
        "--no-resume",
    ]
    if args.disable_physics:
        cmd.append("--disable-physics")
    if args.enforce_depth_valid_max:
        cmd.append("--enforce-depth-valid-max")
    else:
        cmd.append("--no-enforce-depth-valid-max")
    if args.opening_require_right_cap:
        cmd.append("--opening-require-right-cap")
    else:
        cmd.append("--no-opening-require-right-cap")
    if args.enable_origin_finish:
        cmd.append("--enable-origin-finish")
    else:
        cmd.append("--no-enable-origin-finish")
    if args.enable_inside_reanchor:
        cmd.append("--enable-inside-reanchor")
    else:
        cmd.append("--no-enable-inside-reanchor")
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

    # Re-ensure parent in case external cleanup/race removes it between scene runs.
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def main():
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

    parser.add_argument("--target-frames", type=int, default=220)
    parser.add_argument("--path-target-m", type=float, default=55.0)
    parser.add_argument("--path-max-m", type=float, default=55.0)
    parser.add_argument("--grid-res-m", type=float, default=0.35)
    parser.add_argument("--novelty-cap-m", type=float, default=4.0)
    parser.add_argument("--novelty-sample-step-m", type=float, default=0.35)
    parser.add_argument("--lambda-novelty", type=float, default=2.2)
    parser.add_argument("--lambda-frontier", type=float, default=1.0)

    parser.add_argument("--wall-target-dist-m", type=float, default=0.45)
    parser.add_argument("--wall-kp", type=float, default=1.0)
    parser.add_argument("--wall-yaw-limit-deg", type=float, default=14.0)
    parser.add_argument("--turn-yaw-limit-deg", type=float, default=14.0)
    parser.add_argument("--commit-yaw-limit-deg", type=float, default=9.0)
    parser.add_argument("--escape-commit-yaw-limit-deg", type=float, default=9.0)

    parser.add_argument("--room-check-commit-dist-m", type=float, default=2.0)
    parser.add_argument("--escape-commit-dist-m", type=float, default=2.5)
    parser.add_argument("--commit-frames-max", type=int, default=18)
    parser.add_argument("--escape-commit-frames-max", type=int, default=24)

    parser.add_argument("--escape-scan-bins", type=int, default=36)
    parser.add_argument("--escape-scan-top-k", type=int, default=3)
    parser.add_argument("--max-escape-attempts", type=int, default=6)
    parser.add_argument("--escape-cooldown-frames", type=int, default=20)

    parser.add_argument("--stuck-window-frames", type=int, default=30)
    parser.add_argument("--stuck-net-disp-m", type=float, default=0.25)
    parser.add_argument("--stuck-path-m", type=float, default=0.60)
    parser.add_argument("--stuck-new-cells-max", type=int, default=1)
    parser.add_argument("--stuck-near-wall-ratio", type=float, default=0.6)
    parser.add_argument("--zero-move-delta-pos-m", type=float, default=0.02)
    parser.add_argument("--zero-move-recovery-streak", type=int, default=6)
    parser.add_argument("--zero-move-escape-streak", type=int, default=12)
    parser.add_argument("--coverage-stall-new-cells-max", type=int, default=1)
    parser.add_argument("--coverage-stall-escape-streak", type=int, default=30)

    parser.add_argument("--corner-strong-f-m", type=float, default=0.55)
    parser.add_argument("--corner-strong-fl-m", type=float, default=0.35)
    parser.add_argument("--corner-delta-f-m", type=float, default=-0.30)
    parser.add_argument("--corner-delta-fl-m", type=float, default=0.45)
    parser.add_argument("--corner-near-f-m", type=float, default=0.60)
    parser.add_argument("--turn-corner-lock-frames", type=int, default=18)
    parser.add_argument("--corner-backoff-m", type=float, default=0.20)
    parser.add_argument("--corner-sidestep-m", type=float, default=0.20)

    parser.add_argument("--opening-delta-l-m", type=float, default=0.40)
    parser.add_argument("--opening-left-m", type=float, default=1.50)
    parser.add_argument("--opening-fwd-m", type=float, default=1.20)
    parser.add_argument("--opening-votes-needed", type=int, default=3)
    parser.add_argument("--opening-right-cap-m", type=float, default=1.20)
    parser.add_argument("--opening-min-valid-ratio", type=float, default=0.70)
    parser.add_argument("--opening-override-delta-l-m", type=float, default=0.55)
    parser.add_argument("--opening-override-fwd-m", type=float, default=0.90)
    parser.add_argument("--opening-persist-frames", type=int, default=3)
    parser.add_argument("--opening-require-right-cap", dest="opening_require_right_cap", action="store_true", default=False)
    parser.add_argument("--no-opening-require-right-cap", dest="opening_require_right_cap", action="store_false")

    parser.add_argument("--step-nominal-m", type=float, default=0.24)
    parser.add_argument("--step-nominal-wall-m", type=float, default=0.24)
    parser.add_argument("--step-nominal-commit-m", type=float, default=0.35)
    parser.add_argument("--step-nominal-escape-m", type=float, default=0.30)
    parser.add_argument("--step-min-m", type=float, default=0.14)
    parser.add_argument("--step-max-m", type=float, default=0.45)
    parser.add_argument("--step-clearance-scale", type=float, default=0.40)
    parser.add_argument("--step-clearance-scale-wall", type=float, default=0.40)
    parser.add_argument("--step-clearance-scale-commit", type=float, default=0.50)
    parser.add_argument("--step-clearance-scale-escape", type=float, default=0.50)
    parser.add_argument("--step-obstacle-margin-m", type=float, default=0.15)
    parser.add_argument("--anti-freeze-fwd-m", type=float, default=1.00)
    parser.add_argument("--anti-freeze-step-m", type=float, default=0.25)
    parser.add_argument("--step-block-fwd-m", type=float, default=0.30)
    parser.add_argument("--last-move-min-delta-m", type=float, default=0.05)
    parser.add_argument("--black-backoff-step-m", type=float, default=0.25)
    parser.add_argument("--black-backoff-steps-medium", type=int, default=1)
    parser.add_argument("--black-backoff-steps-heavy", type=int, default=2)
    parser.add_argument("--black-recovery-commit-dist-m", type=float, default=2.0)
    parser.add_argument("--black-recovery-commit-min-frames", type=int, default=8)
    parser.add_argument("--black-lateral-primary-deg", type=float, default=90.0)
    parser.add_argument("--black-lateral-secondary-deg", type=float, default=60.0)
    parser.add_argument("--black-lateral-secondary-when-tight-m", type=float, default=0.90)
    parser.add_argument("--no-back-forbid-half-deg", type=float, default=35.0)
    parser.add_argument("--no-back-forbid-frames", type=int, default=40)
    parser.add_argument("--no-back-forbid-score-scale", type=float, default=0.0)
    parser.add_argument("--escape-blocked-rescan-streak", type=int, default=8)

    parser.add_argument("--turn-enter-fwd-m", type=float, default=0.55)
    parser.add_argument("--turn-exit-fwd-m", type=float, default=0.70)
    parser.add_argument("--turn-min-frames", type=int, default=3)
    parser.add_argument("--turn-to-escape-frames", type=int, default=16)

    parser.add_argument("--startup-wall-frames", type=int, default=15)
    parser.add_argument("--startup-wall-path-m", type=float, default=2.0)
    parser.add_argument("--delta-h-max-m", type=float, default=0.08)
    parser.add_argument("--wall-delta-pos-max-m", type=float, default=0.30)
    parser.add_argument("--commit-delta-pos-max-m", type=float, default=0.40)
    parser.add_argument("--turn-delta-pos-max-m", type=float, default=0.10)
    parser.add_argument("--wall-delta-yaw-max-deg", type=float, default=20.0)
    parser.add_argument("--commit-delta-yaw-max-deg", type=float, default=20.0)
    parser.add_argument("--turn-delta-yaw-max-deg", type=float, default=60.0)
    parser.add_argument("--max-consecutive-recovery-fail", type=int, default=8)

    parser.add_argument("--depth-valid-min-m", type=float, default=1e-4)
    parser.add_argument("--depth-valid-max-m", type=float, default=200.0)
    parser.add_argument("--enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_true", default=False)
    parser.add_argument("--no-enforce-depth-valid-max", dest="enforce_depth_valid_max", action="store_false")
    parser.add_argument("--near-wall-depth-m", type=float, default=0.50)
    parser.add_argument("--near-wall-ratio-high", type=float, default=0.35)
    parser.add_argument("--blank-valid-ratio", type=float, default=0.05)
    parser.add_argument("--blank-frame-streak-max", type=int, default=8)
    parser.add_argument("--black-pixel-threshold", type=int, default=5)
    parser.add_argument("--black-ratio-recovery", type=float, default=0.35)
    parser.add_argument("--black-soft-void-ratio", type=float, default=0.30)
    parser.add_argument("--black-soft-ratio", type=float, default=0.35)
    parser.add_argument("--black-ratio-fail", type=float, default=0.25)
    parser.add_argument("--black-fail-streak", type=int, default=2)
    parser.add_argument("--depth-valid-ratio-recovery-min", type=float, default=0.60)
    parser.add_argument("--depth-valid-ratio-recovery-streak", type=int, default=2)
    parser.add_argument("--depth-valid-ratio-fail-streak", type=int, default=8)
    parser.add_argument("--outside-valid-ratio-min", type=float, default=0.50)
    parser.add_argument("--outside-center-valid-ratio-max", type=float, default=0.08)
    parser.add_argument("--outside-side-valid-ratio-max", type=float, default=0.08)
    parser.add_argument("--outside-proxy-recovery-streak", type=int, default=2)
    parser.add_argument("--outside-proxy-reanchor-streak", type=int, default=3)
    parser.add_argument("--outside-proxy-fail-streak", type=int, default=8)
    parser.add_argument("--inside-backoff-m", type=float, default=0.25)
    parser.add_argument("--enable-inside-reanchor", dest="enable_inside_reanchor", action="store_true", default=True)
    parser.add_argument("--no-enable-inside-reanchor", dest="enable_inside_reanchor", action="store_false")
    parser.add_argument("--inside-reanchor-max", type=int, default=4)
    parser.add_argument("--safety-min-depth-m", type=float, default=0.12)
    parser.add_argument("--collision-min-depth-m", type=float, default=0.25)
    parser.add_argument("--blocked-step-min-m", type=float, default=0.12)
    parser.add_argument("--blocked-delta-pos-m", type=float, default=0.03)
    parser.add_argument("--recovery-step-cap-m", type=float, default=0.25)

    parser.add_argument("--escape-success-new-cells-min", type=int, default=1)
    parser.add_argument("--plateau-window-path-m", type=float, default=8.0)
    parser.add_argument("--plateau-new-cells-max", type=int, default=8)
    parser.add_argument("--plateau-min-net-disp-m", type=float, default=1.0)
    parser.add_argument("--plateau-max-escape-attempts-consecutive", type=int, default=2)
    parser.add_argument("--coverage-gain-recent-max", type=int, default=1)
    parser.add_argument("--path-min-done-m", type=float, default=12.0)
    parser.add_argument("--hazard-soft-timeout-path-m", type=float, default=5.0)
    parser.add_argument("--path-coverage-done-min-m", type=float, default=8.0)
    parser.add_argument("--reanchor-delta-pos-max-m", type=float, default=10.0)
    parser.add_argument("--reanchor-delta-yaw-max-deg", type=float, default=180.0)

    parser.add_argument("--step3-pitch-deg", type=float, default=0.0)

    parser.add_argument("--enable-origin-finish", dest="enable_origin_finish", action="store_true", default=False)
    parser.add_argument("--no-enable-origin-finish", dest="enable_origin_finish", action="store_false")
    parser.add_argument("--origin-finish-pos-m", type=float, default=0.60)
    parser.add_argument("--origin-finish-yaw-deg", type=float, default=20.0)
    parser.add_argument("--origin-finish-streak", type=int, default=5)
    parser.add_argument("--export-frames", dest="export_frames", action="store_true", default=True)
    parser.add_argument("--no-export-frames", dest="export_frames", action="store_false")
    parser.add_argument("--make-video", dest="make_video", action="store_true", default=True)
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-fallback-gif", dest="video_fallback_gif", action="store_true", default=True)
    parser.add_argument("--no-video-fallback-gif", dest="video_fallback_gif", action="store_false")

    args = parser.parse_args()

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
        f"Step3 batch done: inventory_total={inventory_total}, newly_processed={processed_new}, "
        f"resume={'on' if args.resume else 'off'}, subprocess_isolation={'on' if args.subprocess_isolation else 'off'}"
    )


if __name__ == "__main__":
    main()
