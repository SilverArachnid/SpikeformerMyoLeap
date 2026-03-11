from spikeformer_myo_leap.config import PreprocessingConfig
from spikeformer_myo_leap.data.manifest import manifest_dataframe


def main():
    config = PreprocessingConfig()
    manifest = manifest_dataframe(config.dataset_root)
    print(f"Found {len(manifest)} episodes under {config.dataset_root}")


if __name__ == "__main__":
    main()
