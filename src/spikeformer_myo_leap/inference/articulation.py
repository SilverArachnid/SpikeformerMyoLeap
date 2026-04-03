"""Canonical biological-hand articulation helpers for live inference.

This module intentionally separates model output interpretation from any
prosthetic-specific mapping. Live checkpoints may predict either:

- 3D point targets
- compact joint-angle targets

Both are converted into the same canonical articulation state so downstream
prosthetic adapters can stay agnostic to the model target representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CANONICAL_ARTICULATION_LABELS: tuple[str, ...] = (
    "thumb_mcp",
    "thumb_ip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
)


@dataclass(frozen=True)
class CanonicalArticulationState:
    """Compact articulation state shared by visualization and retargeting.

    The schema focuses on flexion-like DoFs that are available reliably from the
    current hand representation. Thumb yaw/roll are intentionally omitted for the
    first pass because the immediate goal is finger flexion control.
    """

    thumb_mcp: float
    thumb_ip: float
    index_mcp: float
    index_pip: float
    index_dip: float
    middle_mcp: float
    middle_pip: float
    middle_dip: float
    ring_mcp: float
    ring_pip: float
    ring_dip: float
    pinky_mcp: float
    pinky_pip: float
    pinky_dip: float

    def as_array(self) -> np.ndarray:
        """Return the canonical state in a stable flat order."""

        return np.asarray(
            [
                self.thumb_mcp,
                self.thumb_ip,
                self.index_mcp,
                self.index_pip,
                self.index_dip,
                self.middle_mcp,
                self.middle_pip,
                self.middle_dip,
                self.ring_mcp,
                self.ring_pip,
                self.ring_dip,
                self.pinky_mcp,
                self.pinky_pip,
                self.pinky_dip,
            ],
            dtype=np.float32,
        )


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    """Return a stable unit vector."""

    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at ``b`` formed by ``a-b-c`` in radians."""

    first = _safe_unit(a - b)
    last = _safe_unit(c - b)
    cosine = float(np.clip(np.dot(first, last), -1.0, 1.0))
    return float(np.arccos(cosine))


def points_to_canonical_articulation(prediction: np.ndarray, target_mode: str) -> CanonicalArticulationState:
    """Convert a point-model prediction into canonical articulation angles."""

    if target_mode != "xyz":
        raise ValueError("Canonical articulation extraction from points requires target_mode='xyz'.")

    points = prediction.reshape(21, 3).astype(np.float32)
    return CanonicalArticulationState(
        thumb_mcp=_joint_angle(points[1], points[2], points[3]),
        thumb_ip=_joint_angle(points[2], points[3], points[4]),
        index_mcp=_joint_angle(points[0], points[5], points[6]),
        index_pip=_joint_angle(points[5], points[6], points[7]),
        index_dip=_joint_angle(points[6], points[7], points[8]),
        middle_mcp=_joint_angle(points[0], points[9], points[10]),
        middle_pip=_joint_angle(points[9], points[10], points[11]),
        middle_dip=_joint_angle(points[10], points[11], points[12]),
        ring_mcp=_joint_angle(points[0], points[13], points[14]),
        ring_pip=_joint_angle(points[13], points[14], points[15]),
        ring_dip=_joint_angle(points[14], points[15], points[16]),
        pinky_mcp=_joint_angle(points[0], points[17], points[18]),
        pinky_pip=_joint_angle(points[17], points[18], points[19]),
        pinky_dip=_joint_angle(points[18], points[19], points[20]),
    )


def joint_angles_to_canonical_articulation(prediction: np.ndarray) -> CanonicalArticulationState:
    """Lift the compact 10-angle target representation into the canonical state.

    The current joint-angle training target omits explicit finger DIP terms. For
    prosthetic retargeting we approximate each DIP from the corresponding PIP so
    downstream adapters still receive a complete flexion-oriented state.
    """

    values = prediction.astype(np.float32).tolist()
    return CanonicalArticulationState(
        thumb_mcp=float(values[0]),
        thumb_ip=float(values[1]),
        index_mcp=float(values[2]),
        index_pip=float(values[3]),
        index_dip=float(values[3]),
        middle_mcp=float(values[4]),
        middle_pip=float(values[5]),
        middle_dip=float(values[5]),
        ring_mcp=float(values[6]),
        ring_pip=float(values[7]),
        ring_dip=float(values[7]),
        pinky_mcp=float(values[8]),
        pinky_pip=float(values[9]),
        pinky_dip=float(values[9]),
    )


def format_articulation_status(state: CanonicalArticulationState, *, max_fields: int = 6) -> str:
    """Return a compact status-line summary."""

    parts = []
    for label, value in zip(CANONICAL_ARTICULATION_LABELS, state.as_array().tolist()):
        parts.append(f"{label}: {value:.2f}")
    return " | ".join(parts[:max_fields])
