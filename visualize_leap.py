import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from spikeformer_myo_leap.scripts.visualize_leap import main


if __name__ == "__main__":
    main()
