"""Package entry point for checkpoint evaluation."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from spikeformer_myo_leap.training import build_evaluation_config, evaluate_model

@hydra.main(config_path="../training/conf", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run checkpoint evaluation from Hydra-composed YAML configuration."""

    config = build_evaluation_config(OmegaConf.to_container(cfg, resolve=True))
    summary = evaluate_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
