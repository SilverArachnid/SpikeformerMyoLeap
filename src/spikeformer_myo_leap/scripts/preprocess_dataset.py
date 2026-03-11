"""Package entry point for lightweight preprocessing pipeline checks."""

from spikeformer_myo_leap.config import PreprocessingConfig
from spikeformer_myo_leap.data.manifest import manifest_dataframe


def main() -> None:
    """Build and print a dataset manifest summary for the default preprocessing config.

    The current defaults are:
    - dataset root: ``datasets``
    - target mode: ``xyz``
    - resample rate: ``100 Hz``
    - EMG window size: ``64``
    - wrist-relative pose normalization: enabled
    """

    config = PreprocessingConfig()
    manifest = manifest_dataframe(config.dataset_root)
    print(f"Found {len(manifest)} episodes under {config.dataset_root}")


if __name__ == "__main__":
    main()
