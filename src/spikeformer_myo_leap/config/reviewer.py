"""Configuration objects for the dataset reviewer GUI."""

from dataclasses import dataclass


@dataclass
class DatasetReviewerConfig:
    """Default settings for the dataset reviewer GUI.

    Attributes:
        dataset_root: Root directory to scan for saved episodes. Defaults to ``"datasets"``.
        use_palm_frame_preview: Whether the hand replay panel should render 3D pose in the
            palm-aligned frame by default. Defaults to ``True``.
    """

    dataset_root: str = "datasets"
    use_palm_frame_preview: bool = True
