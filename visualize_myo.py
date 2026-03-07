import time
import argparse
import pyomyo
import rerun as rr

from rerun_viewer import init_rerun

def main():
    parser = argparse.ArgumentParser(description="Live visualize Myo Armband 8-channel EMG data using Rerun.")
    parser.add_argument("--web", action="store_true", help="Force the Rerun web viewer.")
    parser.add_argument("--native", action="store_true", help="Force the native Rerun viewer.")
    args = parser.parse_args()

    use_web = init_rerun("SpikeFormerMyo_VisualizeMyo", force_web=args.web, force_native=args.native)
    if use_web:
        print("Using Rerun web viewer.")
    else:
        print("Using native Rerun viewer.")

    # Initialize Myo
    print("Connecting to Myo...")
    m = pyomyo.Myo(mode=pyomyo.emg_mode.PREPROCESSED)
    m.connect()

    def myo_handler(emg, movement):
        # We get exactly 8 channels
        for i, val in enumerate(emg):
            rr.log(f"emg/ch{i+1}", rr.TimeSeriesScalar(val))

    m.add_emg_handler(myo_handler)
    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    print("Connected. Streaming EMG data to Rerun. Press Ctrl+C to stop.")

    try:
        while True:
            m.run()
    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        print("Disconnected Myo.")

if __name__ == "__main__":
    main()
