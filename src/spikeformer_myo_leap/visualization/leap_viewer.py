import argparse
import time

import leap
import numpy as np
import rerun as rr

from spikeformer_myo_leap.visualization.local_dashboard import LocalDashboard
from spikeformer_myo_leap.visualization.rerun import init_rerun


def extract_hand_points(hand):
    wrist = hand.arm.next_joint
    points = [[wrist.x, wrist.y, wrist.z]]
    for digit in hand.digits:
        points.extend([
            [digit.metacarpal.next_joint.x, digit.metacarpal.next_joint.y, digit.metacarpal.next_joint.z],
            [digit.proximal.next_joint.x, digit.proximal.next_joint.y, digit.proximal.next_joint.z],
            [digit.intermediate.next_joint.x, digit.intermediate.next_joint.y, digit.intermediate.next_joint.z],
            [digit.distal.next_joint.x, digit.distal.next_joint.y, digit.distal.next_joint.z],
        ])
    return np.asarray(points, dtype=np.float32)


class BaseTrackerListener(leap.Listener):
    def on_connection_event(self, event):
        print("Connected to Leap Service.")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
                print(f"Found device: {info.serial}")
        except leap.LeapCannotOpenDeviceError:
            print("Cannot open Leap device.")


class LocalTrackerListener(BaseTrackerListener):
    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def on_tracking_event(self, event):
        hands = [extract_hand_points(hand) for hand in event.hands]
        self.dashboard.update_hand(hands)
        self.dashboard.update_status(
            mode="Leap Preview",
            status_line="Streaming local Leap hand pose viewer",
        )


class RerunTrackerListener(BaseTrackerListener):
    def on_tracking_event(self, event):
        digit_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        digit_colors = {
            "Thumb": [255, 0, 0],
            "Index": [0, 255, 0],
            "Middle": [0, 0, 255],
            "Ring": [255, 255, 0],
            "Pinky": [255, 0, 255],
        }

        for hand in event.hands:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            prefix = f"hand/{hand_type}_{hand.id}"
            wrist = hand.arm.next_joint
            points = [[wrist.x, wrist.y, wrist.z]]
            colors = [[255, 255, 255]]
            labels = ["Wrist"]
            edges = []

            for digit_idx, digit in enumerate(hand.digits):
                digit_name = digit_names[digit_idx] if digit_idx < len(digit_names) else f"Digit{digit_idx}"
                color = digit_colors.get(digit_name, [200, 200, 200])
                strip = [[wrist.x, wrist.y, wrist.z]]
                for joint_idx, joint in enumerate(
                    [digit.metacarpal.next_joint, digit.proximal.next_joint, digit.intermediate.next_joint, digit.distal.next_joint]
                ):
                    points.append([joint.x, joint.y, joint.z])
                    colors.append(color)
                    labels.append(f"{digit_name}_{joint_idx}")
                    strip.append([joint.x, joint.y, joint.z])
                edges.append(strip)

            rr.log(f"{prefix}/joints", rr.Points3D(points, colors=colors, labels=labels, radii=5.0))
            rr.log(f"{prefix}/skeleton", rr.LineStrips3D(edges, colors=[[150, 150, 150]] * 5))


def main():
    parser = argparse.ArgumentParser(description="Live visualize Leap Motion tracking.")
    parser.add_argument("--viewer", choices=["local", "rerun"], default="local", help="Visualization backend.")
    parser.add_argument("--web", action="store_true", help="Force the Rerun web viewer.")
    parser.add_argument("--native", action="store_true", help="Force the native Rerun viewer.")
    args = parser.parse_args()

    dashboard = None
    if args.viewer == "rerun":
        use_web = init_rerun("SpikeFormerMyo_VisualizeLeap", force_web=args.web, force_native=args.native)
        print(f"Using Rerun {'web' if use_web else 'native'} viewer.")
        listener = RerunTrackerListener()
    else:
        dashboard = LocalDashboard("SpikeformerMyoLeap  |  Leap Visualizer", show_hand=True, show_emg=False)
        dashboard.start()
        dashboard.update_status(
            mode="Leap Preview",
            recording=False,
            pose_name="-",
            subject_id="-",
            episode_label="Preview",
            status_line="Initializing local dark-mode viewer",
        )
        print("Using local dark-mode Leap viewer.")
        listener = LocalTrackerListener(dashboard)

    connection = leap.Connection()
    connection.add_listener(listener)

    print("Connecting to Leap Service...")
    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            print("Connected. Streaming Leap tracking. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard is not None:
            dashboard.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
