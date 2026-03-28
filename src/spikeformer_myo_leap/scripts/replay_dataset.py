"""Launch the dataset reviewer GUI with Hydra-backed config."""

import hydra
from omegaconf import DictConfig

from spikeformer_myo_leap.app.dataset_reviewer import main as reviewer_main
from spikeformer_myo_leap.config import DatasetReviewerConfig


@hydra.main(config_path="conf", config_name="replay_dataset", version_base="1.3")
def main(cfg: DictConfig):
    reviewer_main(
        DatasetReviewerConfig(
            dataset_root=str(cfg.dataset_root),
            use_palm_frame_preview=bool(cfg.use_palm_frame_preview),
        )
    )


if __name__ == "__main__":
    main()
