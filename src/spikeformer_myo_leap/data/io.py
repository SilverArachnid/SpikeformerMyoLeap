import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .contracts import LANDMARK_NAMES


def session_dir(save_dir, subject_id, session_name):
    return os.path.join(save_dir, subject_id, session_name)


def pose_dir(save_dir, subject_id, session_name, pose_name):
    return os.path.join(session_dir(save_dir, subject_id, session_name), pose_name)


def next_episode_number(save_dir, subject_id, session_name, pose_name):
    root = pose_dir(save_dir, subject_id, session_name, pose_name)
    if not os.path.isdir(root):
        return 1
    existing = [
        name for name in os.listdir(root)
        if name.startswith("ep_") and os.path.isdir(os.path.join(root, name))
    ]
    return len(existing) + 1


def episode_dir(save_dir, subject_id, session_name, pose_name, episode_number):
    return os.path.join(pose_dir(save_dir, subject_id, session_name, pose_name), f"ep_{episode_number:04d}")


def save_episode(
    *,
    settings,
    episode_number,
    emg_data,
    pose_data,
    episode_id,
    recorded_duration_seconds,
):
    folder = episode_dir(
        settings.save_dir,
        settings.subject_id,
        settings.session_name,
        settings.pose_name,
        episode_number,
    )
    os.makedirs(folder, exist_ok=False)

    if emg_data:
        ts_emg, *channels = zip(*emg_data)
        emg_arr = np.array(channels).T
        emg_df = pd.DataFrame(emg_arr, columns=[f"Channel_{i+1}" for i in range(8)])
        emg_df.insert(0, "Timestamp_ms", ts_emg)
        emg_df.to_csv(os.path.join(folder, "emg.csv"), index=False)

    if pose_data:
        ts_pose, *pose_pts = zip(*pose_data)
        pose_arr = np.array(pose_pts).T
        columns = []
        for name in LANDMARK_NAMES:
            columns.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])
        pose_df = pd.DataFrame(pose_arr, columns=columns)
        pose_df.insert(0, "Timestamp_ms", ts_pose)
        pose_df.to_csv(os.path.join(folder, "pose.csv"), index=False)

    effective_emg_hz = 0.0 if recorded_duration_seconds <= 0.0 else len(emg_data) / recorded_duration_seconds
    effective_pose_hz = 0.0 if recorded_duration_seconds <= 0.0 else len(pose_data) / recorded_duration_seconds
    metadata = {
        "episode_id": episode_id,
        "subject_id": settings.subject_id,
        "session_name": settings.session_name,
        "pose_name": settings.pose_name,
        "episode_number": episode_number,
        "duration_seconds": settings.episode_duration,
        "recorded_duration_seconds": recorded_duration_seconds,
        "sample_count_emg": len(emg_data),
        "sample_count_pose": len(pose_data),
        "effective_emg_hz": effective_emg_hz,
        "effective_pose_hz": effective_pose_hz,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "save_root": os.path.abspath(settings.save_dir),
        "session_dir": session_dir(settings.save_dir, settings.subject_id, settings.session_name),
    }
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return folder
