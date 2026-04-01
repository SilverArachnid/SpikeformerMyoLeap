"""Package entry point for checkpoint-backed live Myo inference."""

import hydra
from omegaconf import DictConfig

from spikeformer_myo_leap.config import LiveInferenceConfig
from spikeformer_myo_leap.inference import run_live_inference


@hydra.main(config_path="conf", config_name="live_inference", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run live Myo inference from Hydra-composed YAML config."""

    run_live_inference(
        LiveInferenceConfig(
            checkpoint_path=str(cfg.checkpoint_path),
            device=str(cfg.device),
            viewer=str(cfg.viewer),
            update_hz=float(cfg.update_hz),
            emg_history_seconds=float(cfg.emg_history_seconds),
            prosthetic_model=str(cfg.prosthetic_model),
            simulator_backend=str(cfg.simulator_backend),
            simulator_model_path=str(cfg.simulator_model_path),
            show_emg=bool(cfg.show_emg),
        )
    )


if __name__ == "__main__":
    main()
