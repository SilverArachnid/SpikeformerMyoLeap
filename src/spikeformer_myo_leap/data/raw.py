from dataclasses import dataclass
import json
import os

import pandas as pd


@dataclass
class EpisodePaths:
    root: str
    emg_csv: str
    pose_csv: str
    meta_json: str


def list_episode_paths(dataset_root):
    episodes = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        if {"emg.csv", "pose.csv", "meta.json"}.issubset(set(filenames)):
            episodes.append(
                EpisodePaths(
                    root=dirpath,
                    emg_csv=os.path.join(dirpath, "emg.csv"),
                    pose_csv=os.path.join(dirpath, "pose.csv"),
                    meta_json=os.path.join(dirpath, "meta.json"),
                )
            )
    episodes.sort(key=lambda item: item.root)
    return episodes


def load_episode_metadata(meta_json_path):
    with open(meta_json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episode(episode_paths):
    return {
        "paths": episode_paths,
        "meta": load_episode_metadata(episode_paths.meta_json),
        "emg": pd.read_csv(episode_paths.emg_csv),
        "pose": pd.read_csv(episode_paths.pose_csv),
    }
