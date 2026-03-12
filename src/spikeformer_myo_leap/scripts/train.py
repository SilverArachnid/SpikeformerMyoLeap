"""Package entry point for model training."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from spikeformer_myo_leap.training import build_training_config, train_model

@hydra.main(config_path="../training/conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run model training from Hydra-composed YAML configuration."""

    config = build_training_config(OmegaConf.to_container(cfg, resolve=True))
    summary = train_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
