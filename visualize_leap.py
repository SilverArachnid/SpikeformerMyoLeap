import argparse
import threading
import time

import leap
import numpy as np
import rerun as rr

from rerun_viewer import init_rerun


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


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
                digit_points = [
                    digit.metacarpal.next_joint,
                    digit.proximal.next_joint,
                    digit.intermediate.next_joint,
                    digit.distal.next_joint,
                ]
                digit_name = digit_names[digit_idx] if digit_idx < len(digit_names) else f"Digit{digit_idx}"
                color = digit_colors.get(digit_name, [200, 200, 200])
                strip = [[wrist.x, wrist.y, wrist.z]]

                for joint_idx, joint in enumerate(digit_points):
                    points.append([joint.x, joint.y, joint.z])
                    colors.append(color)
                    labels.append(f"{digit_name}_{joint_idx}")
                    strip.append([joint.x, joint.y, joint.z])

                edges.append(strip)

            rr.log(f"{prefix}/joints", rr.Points3D(points, colors=colors, labels=labels, radii=5.0))
            rr.log(f"{prefix}/skeleton", rr.LineStrips3D(edges, colors=[[150, 150, 150]] * 5))


class MatplotlibTrackerListener(BaseTrackerListener):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest_hands = []

    def on_tracking_event(self, event):
        hands = [extract_hand_points(hand) for hand in event.hands]
        with self._lock:
            self._latest_hands = hands

    def get_latest_hands(self):
        with self._lock:
            return [hand.copy() for hand in self._latest_hands]


def run_matplotlib_viewer(listener):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(_frame):
        ax.clear()
        ax.set_title("Leap Hand Tracking")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-250, 250)
        ax.set_ylim(50, 450)
        ax.set_zlim(-250, 250)
        ax.view_init(elev=20, azim=-70)

        hands = listener.get_latest_hands()
        colors = ["tab:blue", "tab:orange"]
        for hand_idx, points in enumerate(hands):
            color = colors[hand_idx % len(colors)]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=25)
            for start_idx, end_idx in HAND_CONNECTIONS:
                ax.plot(
                    [points[start_idx, 0], points[end_idx, 0]],
                    [points[start_idx, 1], points[end_idx, 1]],
                    [points[start_idx, 2], points[end_idx, 2]],
                    color=color,
                    linewidth=2,
                )

    animation = FuncAnimation(fig, update, interval=33, cache_frame_data=False)
    print("Using local Matplotlib viewer.")
    print("Close the plot window or press Ctrl+C in the terminal to stop.")
    try:
        plt.show()
    finally:
        del animation


def main():
    parser = argparse.ArgumentParser(description="Live visualize Leap Motion tracking.")
    parser.add_argument(
        "--viewer",
        choices=["matplotlib", "rerun"],
        default="matplotlib",
        help="Viewer backend to use. 'matplotlib' is the reliable local desktop option.",
    )
    parser.add_argument("--web", action="store_true", help="Force the Rerun web viewer.")
    parser.add_argument("--native", action="store_true", help="Force the native Rerun viewer.")
    args = parser.parse_args()

    if args.viewer == "rerun":
        use_web = init_rerun("SpikeFormerMyo_VisualizeLeap", force_web=args.web, force_native=args.native)
        print(f"Using Rerun {'web' if use_web else 'native'} viewer.")
        listener = RerunTrackerListener()
    else:
        listener = MatplotlibTrackerListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    print("Connecting to Leap Service...")
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        print("Connected. Streaming Leap tracking. Press Ctrl+C to stop.")
        if args.viewer == "matplotlib":
            run_matplotlib_viewer(listener)
            print("Disconnected.")
        else:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                print("Disconnected.")


if __name__ == "__main__":
    main()
