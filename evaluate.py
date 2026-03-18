"""Compatibility wrapper for the packaged evaluation entry point."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spikeformer_myo_leap.scripts.evaluate import main


if __name__ == "__main__":
    main()
