import argparse
import csv
import glob
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def load_matrix(path):
    mat = np.loadtxt(path)
    if mat.shape != (4, 4):
        raise ValueError(f"{path} is not a 4x4 matrix")
    return mat.astype(np.float64)


def sorted_pose_files(pred_dir):
    files = []
    for path in glob.glob(os.path.join(pred_dir, "*.txt")):
        stem = Path(path).stem
        try:
            stamp = float(stem)
        except ValueError:
            continue
        files.append((stamp, path))
    return sorted(files, key=lambda item: item[0])


def nearest_gt_indices(pred_ts, gt_ts, max_diff=None):
    indices = []
    diffs = []
    gt_ts = np.asarray(gt_ts)
    for ts in pred_ts:
        idx = int(np.argmin(np.abs(gt_ts - ts)))
        diff = float(abs(gt_ts[idx] - ts))
        if max_diff is not None and diff > max_diff:
            indices.append(None)
        else:
            indices.append(idx)
        diffs.append(diff)
    return indices, diffs


def umeyama_alignment(src, dst, with_scale=True):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    cov = dst_centered.T @ src_centered / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1, -1] = -1
    rot = u @ s @ vt
    scale = 1.0
    if with_scale:
        var = np.sum(src_centered ** 2) / src.shape[0]
        if var > 0:
            scale = float(np.trace(np.diag(d) @ s) / var)
    trans = dst_mean - scale * rot @ src_mean
    return scale, rot, trans


def apply_sim3_to_poses(poses, scale, rot, trans):
    aligned = poses.copy()
    aligned[:, :3, :3] = rot[None, :, :] @ poses[:, :3, :3]
    aligned[:, :3, 3] = scale * (poses[:, :3, 3] @ rot.T) + trans
    return aligned


def rotation_angle_deg(rot):
    value = (np.trace(rot) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def evaluate_trajectory(output_dir):
    config_path = os.path.join(output_dir, "config.yaml")
    pred_dir = os.path.join(output_dir, "droid_c2w")
    if not os.path.isfile(config_path) or not os.path.isdir(pred_dir):
        return {"status": "missing_config_or_pred"}

    pose_files = sorted_pose_files(pred_dir)
    if len(pose_files) < 3:
        return {"status": "too_few_pred_poses", "pred_poses": len(pose_files)}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        gt = load_gt_from_config(cfg)
    except Exception as exc:
        return {"status": "gt_load_failed", "error": repr(exc), "pred_poses": len(pose_files)}

    pred_ts = np.array([item[0] for item in pose_files], dtype=np.float64)
    pred_poses = np.array([load_matrix(path) for _, path in pose_files])
    gt_ts = np.asarray(gt["timestamps"], dtype=np.float64)
    gt_poses_all = np.asarray(gt["c2ws"], dtype=np.float64)
    gt_indices, diffs = nearest_gt_indices(pred_ts, gt_ts)
    valid = [i for i, idx in enumerate(gt_indices) if idx is not None]
    if len(valid) < 3:
        return {"status": "too_few_matched_poses", "pred_poses": len(pose_files)}

    pred_poses = pred_poses[valid]
    matched_gt = np.array([gt_poses_all[gt_indices[i]] for i in valid])
    matched_ts = pred_ts[valid]
    matched_diffs = np.array([diffs[i] for i in valid], dtype=np.float64)

    align = umeyama_alignment(pred_poses[:, :3, 3], matched_gt[:, :3, 3], with_scale=True)
    if align is None:
        return {"status": "alignment_failed", "pred_poses": len(pose_files)}
    scale, rot, trans = align
    aligned_pred = apply_sim3_to_poses(pred_poses, scale, rot, trans)
    trans_err = np.linalg.norm(aligned_pred[:, :3, 3] - matched_gt[:, :3, 3], axis=1)

    result = {
        "status": "ok",
        "pred_poses": len(pose_files),
        "matched_poses": int(len(matched_gt)),
        "timestamp_mean_abs_diff": float(matched_diffs.mean()),
        "sim3_scale": float(scale),
        "ate_rmse_m": float(np.sqrt(np.mean(trans_err ** 2))),
        "ate_mean_m": float(np.mean(trans_err)),
        "ate_median_m": float(np.median(trans_err)),
        "ate_max_m": float(np.max(trans_err)),
    }

    rpe = evaluate_kitti_rpe(aligned_pred, matched_gt, matched_ts)
    result.update(rpe)
    return result


def load_gt_from_config(cfg):
    root = cfg["dataset"]["root"]
    module = cfg["dataset"]["module"]

    pose_dir = os.path.join(root, "pose")
    if os.path.isdir(pose_dir):
        c2w_files = [name for name in os.listdir(pose_dir) if name.endswith(".txt")]
        c2ws = np.array([np.loadtxt(os.path.join(pose_dir, name)) for name in c2w_files], dtype=np.float64)
        timestamps = np.array([float(name.replace(".txt", "").split(".")[0]) for name in c2w_files], dtype=np.float64)
        order = np.argsort(timestamps)
        return {"timestamps": timestamps[order], "c2ws": c2ws[order]}

    kitti360_gt = os.path.join(root, "metadata", "gt_local.txt")
    if "kitti360" in module and os.path.isfile(kitti360_gt):
        from lietorch import SE3
        import torch

        timestamp_tqs = np.loadtxt(kitti360_gt)
        timestamps = timestamp_tqs[:, 0].astype(np.float64)
        c2ws = SE3(torch.tensor(timestamp_tqs[:, 1:])).matrix().numpy().astype(np.float64)
        return {"timestamps": timestamps, "c2ws": c2ws}

    oxts_dir = os.path.join(root, "oxts", "data")
    camstamp_path = os.path.join(root, "metadata", "camstamp.txt")
    c2i_path = os.path.join(root, "metadata", "c2i.txt")
    if "kitti_sync" in module and os.path.isdir(oxts_dir) and os.path.isfile(camstamp_path):
        from datasets.pykitti_unsync.pykitti_utils import load_oxts_packets_and_poses

        oxts_files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
        oxts = load_oxts_packets_and_poses(oxts_files)
        imu_poses = np.array([item.T_w_imu for item in oxts], dtype=np.float64)
        c2i = np.loadtxt(c2i_path).astype(np.float64) if os.path.isfile(c2i_path) else np.eye(4)
        c2ws = imu_poses @ c2i
        timestamps = np.loadtxt(camstamp_path, dtype=str)[:, 0].astype(np.float64)
        count = min(len(timestamps), len(c2ws))
        return {"timestamps": timestamps[:count], "c2ws": c2ws[:count]}

    get_dataset = importlib.import_module(module).get_dataset
    dataset = get_dataset(cfg)
    return dataset.load_gt_dict()


def trajectory_distances(poses):
    dist = [0.0]
    for i in range(1, len(poses)):
        step = np.linalg.norm(poses[i, :3, 3] - poses[i - 1, :3, 3])
        dist.append(dist[-1] + float(step))
    return np.array(dist, dtype=np.float64)


def last_frame_from_segment_length(dist, first, length):
    target = dist[first] + length
    idx = np.searchsorted(dist, target)
    if idx < len(dist):
        return int(idx)
    return -1


def evaluate_kitti_rpe(pred, gt, ts):
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    dist = trajectory_distances(gt)
    if dist[-1] < 1.0:
        return {"rpe_status": "trajectory_too_short"}

    trans_errors = []
    rot_errors = []
    for first in range(0, len(gt), 10):
        for length in lengths:
            last = last_frame_from_segment_length(dist, first, length)
            if last < 0:
                continue
            gt_delta = np.linalg.inv(gt[first]) @ gt[last]
            pred_delta = np.linalg.inv(pred[first]) @ pred[last]
            err = np.linalg.inv(pred_delta) @ gt_delta
            trans_errors.append(np.linalg.norm(err[:3, 3]) / length)
            rot_errors.append(rotation_angle_deg(err[:3, :3]) / length)

    if not trans_errors:
        return {
            "rpe_status": "no_valid_segments",
            "trajectory_length_m": float(dist[-1]),
        }
    return {
        "rpe_status": "ok",
        "trajectory_length_m": float(dist[-1]),
        "tREL_percent": float(np.mean(trans_errors) * 100.0),
        "rREL_deg_per_100m": float(np.mean(rot_errors) * 100.0),
        "rpe_segments": int(len(trans_errors)),
    }


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim_score(img1, img2):
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(img1, img2, channel_axis=2, data_range=1.0))
    except Exception:
        return float("nan")


def init_lpips(device):
    try:
        import lpips
        import torch

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model, torch
    except Exception as exc:
        return None, repr(exc)


def lpips_score(model, torch_mod, img1, img2, device):
    if model is None:
        return float("nan")
    arr1 = torch_mod.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float()
    arr2 = torch_mod.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float()
    arr1 = arr1.to(device) * 2.0 - 1.0
    arr2 = arr2.to(device) * 2.0 - 1.0
    with torch_mod.no_grad():
        return float(model(arr1, arr2).item())


def crop_rgbdnua(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = image.shape[:2]
    tile_h = h // 2
    tile_w = w // 4
    gt_rgb = image[:tile_h, :tile_w, :]
    pred_rgb = image[tile_h : 2 * tile_h, :tile_w, :]
    return pred_rgb, gt_rgb


def evaluate_render_from_rgbdnua(output_dir, device="cpu"):
    rgbdnua_dir = os.path.join(output_dir, "rgbdnua")
    files = sorted(glob.glob(os.path.join(rgbdnua_dir, "*.png")))
    if not files:
        return {"status": "no_rgbdnua_png", "frames": 0}

    lpips_model, torch_or_error = init_lpips(device)
    lpips_error = None if lpips_model is not None else str(torch_or_error)
    torch_mod = torch_or_error if lpips_model is not None else None

    psnrs = []
    ssims = []
    lpipss = []
    used = 0
    failed = 0
    for path in files:
        try:
            pred, gt = crop_rgbdnua(path)
            psnrs.append(psnr(pred, gt))
            ssims.append(ssim_score(pred, gt))
            if lpips_model is not None:
                lpipss.append(lpips_score(lpips_model, torch_mod, pred, gt, device))
            used += 1
        except Exception:
            failed += 1

    result = {
        "status": "ok" if used else "all_failed",
        "source": "rgbdnua_cropped_png",
        "frames": int(used),
        "failed_frames": int(failed),
        "psnr_db": float(np.mean(psnrs)) if psnrs else float("nan"),
        "ssim": float(np.nanmean(ssims)) if ssims else float("nan"),
        "lpips": float(np.mean(lpipss)) if lpipss else float("nan"),
    }
    if lpips_error:
        result["lpips_error"] = lpips_error
    return result


def parse_runtime(log_path):
    result = {}
    if not log_path or not os.path.isfile(log_path):
        return result
    text = Path(log_path).read_text(errors="ignore")
    result["log_path"] = log_path
    result["oom"] = "OutOfMemoryError" in text or "CUDA out of memory" in text
    progress = re.findall(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<", text)
    if progress:
        pct, done, total, elapsed = progress[-1]
        result.update(
            {
                "progress_percent": int(pct),
                "frames_done": int(done),
                "frames_total": int(total),
                "elapsed": elapsed,
            }
        )
    gauss_gpu = re.findall(r"Gaussian num on GPU:\s*(\d+)", text)
    gauss_cpu = re.findall(r"Gaussian num on CPU:\s*(\d+)", text)
    if gauss_gpu:
        result["gaussian_gpu_last"] = int(gauss_gpu[-1])
        result["gaussian_gpu_max"] = int(max(map(int, gauss_gpu)))
    if gauss_cpu:
        result["gaussian_cpu_last"] = int(gauss_cpu[-1])
        result["gaussian_cpu_max"] = int(max(map(int, gauss_cpu)))
    return result


def ply_info(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "ply", "idx=*.ply")))
    if not files:
        return {"ply_path": "", "model_size_mb": float("nan")}
    path = files[-1]
    return {"ply_path": path, "model_size_mb": os.path.getsize(path) / (1024.0 * 1024.0)}


def flatten(prefix, data, out):
    for key, value in data.items():
        out[f"{prefix}_{key}"] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", nargs=3, metavar=("NAME", "OUTPUT_DIR", "LOG_PATH"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    rows = []
    details = {}
    for name, output_dir, log_path in args.case:
        output_dir = os.path.abspath(output_dir)
        row = {"case": name, "output_dir": output_dir}
        traj = evaluate_trajectory(output_dir)
        render = {"status": "skipped"} if args.no_render else evaluate_render_from_rgbdnua(output_dir, args.device)
        runtime = parse_runtime(log_path)
        ply = ply_info(output_dir)
        flatten("traj", traj, row)
        flatten("render", render, row)
        flatten("runtime", runtime, row)
        row.update(ply)
        rows.append(row)
        details[name] = {"trajectory": traj, "render": render, "runtime": runtime, "ply": ply}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    fields = sorted({key for row in rows for key in row.keys()})
    with open(out_path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps(details, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
