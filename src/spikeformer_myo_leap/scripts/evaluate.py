"""Package entry point for checkpoint evaluation."""

from __future__ import annotations

import json

from spikeformer_myo_leap.training import EvaluationConfig, evaluate_model


def main() -> None:
    """Run evaluation with the default packaged evaluation config."""

    config = EvaluationConfig()
    if not config.checkpoint_path:
        raise ValueError(
            "Default evaluation config has no checkpoint_path set. "
            "Construct EvaluationConfig(checkpoint_path=...) before using this entry point."
        )
    summary = evaluate_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
