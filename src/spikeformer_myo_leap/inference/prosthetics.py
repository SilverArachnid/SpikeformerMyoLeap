"""Prosthetic retargeting adapters for live inference.

The first pass deliberately stops at joint-command generation. MuJoCo or any
other simulator backend can consume these adapter outputs later.
"""

from __future__ import annotations

from dataclasses import dataclass

from spikeformer_myo_leap.inference.articulation import CanonicalArticulationState


SUPPORTED_PROSTHETIC_MODELS: tuple[str, ...] = (
    "none",
    "ability_hand",
    "dexhandv2_right",
    "dexhandv2_cobot_right",
)


@dataclass(frozen=True)
class ProstheticCommandSet:
    """Adapter output for one prosthetic-hand target."""

    model_name: str
    joint_targets: dict[str, float]
    note: str | None = None


def _map_ability_hand(state: CanonicalArticulationState) -> ProstheticCommandSet:
    return ProstheticCommandSet(
        model_name="ability_hand",
        joint_targets={
            "thumb_q1": state.thumb_mcp,
            "thumb_q2": state.thumb_ip,
            "index_q1": state.index_mcp,
            "index_q2": state.index_pip,
            "middle_q1": state.middle_mcp,
            "middle_q2": state.middle_pip,
            "ring_q1": state.ring_mcp,
            "ring_q2": state.ring_pip,
            "pinky_q1": state.pinky_mcp,
            "pinky_q2": state.pinky_pip,
        },
        note="Ability Hand q2 commands absorb distal finger flexion; DIP is not driven independently.",
    )


def _map_dexhand(state: CanonicalArticulationState, *, model_name: str) -> ProstheticCommandSet:
    return ProstheticCommandSet(
        model_name=model_name,
        joint_targets={
            "R_Thumb_Yaw": 0.0,
            "R_Thumb_Roll": 0.0,
            "R_Thumb_Pitch": state.thumb_mcp,
            "R_Thumb_Flexor": state.thumb_ip,
            "R_Thumb_DIP": state.thumb_ip,
            "R_Index_Yaw": 0.0,
            "R_Index_Pitch": state.index_mcp,
            "R_Index_Flexor": state.index_pip,
            "R_Index_DIP": state.index_dip,
            "R_Middle_Pitch": state.middle_mcp,
            "R_Middle_Flexor": state.middle_pip,
            "R_Middle_DIP": state.middle_dip,
            "R_Ring_Pitch": state.ring_mcp,
            "R_Ring_Flexor": state.ring_pip,
            "R_Ring_DIP": state.ring_dip,
            "R_Pinky_Pitch": state.pinky_mcp,
            "R_Pinky_Flexor": state.pinky_pip,
            "R_Pinky_DIP": state.pinky_dip,
        },
        note="Thumb yaw/roll and index yaw are held at zero in the first pass.",
    )


def map_articulation_to_prosthetic(
    state: CanonicalArticulationState,
    model_name: str,
) -> ProstheticCommandSet | None:
    """Map canonical articulation into one supported prosthetic target."""

    if model_name == "none":
        return None
    if model_name == "ability_hand":
        return _map_ability_hand(state)
    if model_name in {"dexhandv2_right", "dexhandv2_cobot_right"}:
        return _map_dexhand(state, model_name=model_name)
    raise ValueError(
        f"Unsupported prosthetic_model={model_name!r}. "
        f"Supported values: {', '.join(SUPPORTED_PROSTHETIC_MODELS)}"
    )


def format_prosthetic_status(commands: ProstheticCommandSet, *, max_fields: int = 5) -> str:
    """Return a compact prosthetic-command summary."""

    parts = []
    for joint_name, value in commands.joint_targets.items():
        parts.append(f"{joint_name}: {value:.2f}")
    summary = " | ".join(parts[:max_fields])
    return f"{commands.model_name} | {summary}"
