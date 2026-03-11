from dataclasses import asdict, dataclass
import os

import pandas as pd

from .raw import list_episode_paths, load_episode_metadata


@dataclass
class EpisodeManifestRecord:
    episode_dir: str
    subject_id: str
    session_name: str
    pose_name: str
    episode_number: int
    sample_count_emg: int
    sample_count_pose: int
    effective_emg_hz: float
    effective_pose_hz: float
    recorded_duration_seconds: float


def build_manifest(dataset_root: str) -> list[EpisodeManifestRecord]:
    records: list[EpisodeManifestRecord] = []
    for episode in list_episode_paths(dataset_root):
        meta = load_episode_metadata(episode.meta_json)
        records.append(
            EpisodeManifestRecord(
                episode_dir=os.path.abspath(episode.root),
                subject_id=str(meta.get("subject_id", "")),
                session_name=str(meta.get("session_name", "")),
                pose_name=str(meta.get("pose_name", "")),
                episode_number=int(meta.get("episode_number", 0)),
                sample_count_emg=int(meta.get("sample_count_emg", 0)),
                sample_count_pose=int(meta.get("sample_count_pose", 0)),
                effective_emg_hz=float(meta.get("effective_emg_hz", 0.0)),
                effective_pose_hz=float(meta.get("effective_pose_hz", 0.0)),
                recorded_duration_seconds=float(meta.get("recorded_duration_seconds", 0.0)),
            )
        )
    records.sort(key=lambda record: record.episode_dir)
    return records


def manifest_dataframe(dataset_root: str) -> pd.DataFrame:
    return pd.DataFrame(asdict(record) for record in build_manifest(dataset_root))
