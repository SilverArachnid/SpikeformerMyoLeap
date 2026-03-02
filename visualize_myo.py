import time
import argparse
import pyomyo
import rerun as rr

def main():
    parser = argparse.ArgumentParser(description="Live visualize Myo Armband 8-channel EMG data using Rerun.")
    parser.add_argument("--web", action="store_true", help="Launch the Rerun web viewer instead of the native viewer to avoid Linux graphics issues.")
    args = parser.parse_args()

    # Initialize Rerun
    rr.init("SpikeFormerMyo_VisualizeMyo")
    if args.web:
        rr.serve_web_viewer()
    else:
        rr.spawn()

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
