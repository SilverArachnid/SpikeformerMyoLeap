import argparse
import time

import pyomyo
import rerun as rr

from local_visualizer import LocalDashboard
from rerun_viewer import init_rerun


def main():
    parser = argparse.ArgumentParser(description="Live visualize Myo Armband 8-channel EMG data.")
    parser.add_argument("--viewer", choices=["local", "rerun"], default="local", help="Visualization backend.")
    parser.add_argument("--web", action="store_true", help="Force the Rerun web viewer.")
    parser.add_argument("--native", action="store_true", help="Force the native Rerun viewer.")
    args = parser.parse_args()

    dashboard = None
    if args.viewer == "rerun":
        use_web = init_rerun("SpikeFormerMyo_VisualizeMyo", force_web=args.web, force_native=args.native)
        print(f"Using Rerun {'web' if use_web else 'native'} viewer.")
    else:
        dashboard = LocalDashboard("SpikeformerMyoLeap  |  Myo Visualizer", show_hand=False, show_emg=True)
        dashboard.start()
        dashboard.update_status(
            mode="Myo Preview",
            recording=False,
            pose_name="-",
            subject_id="-",
            episode_label="Preview",
            status_line="Initializing local EMG dashboard",
        )
        print("Using local dark-mode Myo viewer.")

    print("Connecting to Myo...")
    m = pyomyo.Myo(mode=pyomyo.emg_mode.PREPROCESSED)
    m.connect()

    emg_count = 0

    def myo_handler(emg, movement):
        nonlocal emg_count
        emg_count += 1
        if dashboard is not None:
            dashboard.update_emg(emg)
            dashboard.update_status(
                mode="Myo Preview",
                sample_count_emg=emg_count,
                sample_count_pose=0,
                status_line="Streaming 8-channel preprocessed EMG",
            )
        else:
            for i, val in enumerate(emg):
                rr.log(f"emg/ch{i+1}", rr.TimeSeriesScalar(val))

    m.add_emg_handler(myo_handler)
    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    print("Connected. Streaming EMG data. Press Ctrl+C to stop.")

    try:
        while True:
            m.run()
    except KeyboardInterrupt:
        pass
    finally:
        m.disconnect()
        if dashboard is not None:
            dashboard.close()
        print("Disconnected Myo.")


if __name__ == "__main__":
    main()
