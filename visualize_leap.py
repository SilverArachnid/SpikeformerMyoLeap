import time
import argparse
import leap
import rerun as rr

class RerunTrackerListener(leap.Listener):
    def on_connection_event(self, event):
        print("Connected to Leap Service.")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
                print(f"Found device: {info.serial}")
        except leap.LeapCannotOpenDeviceError:
            print("Cannot open Leap device.")

    def on_tracking_event(self, event):
        # We will visualize all hands found
        for hand in event.hands:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            prefix = f"hand/{hand_type}_{hand.id}"

            # 1. Wrist
            wrist = hand.arm.next_joint
            points = [[wrist.x, wrist.y, wrist.z]]
            colors = [[255, 255, 255]] # Wrist = white
            labels = ["Wrist"]

            # Connection pairs for LineStrips3D
            edges = []

            # 2. Digits (Thumb, Index, Middle, Ring, Pinky)
            for digit in hand.digits:
                digit_points = [
                    digit.metacarpal.next_joint,
                    digit.proximal.next_joint,
                    digit.intermediate.next_joint,
                    digit.distal.next_joint,
                ]

                # Map to standard Rerun color based on digit type
                digit_type_str = str(digit.finger_type) # e.g., "FingerType.Thumb"
                cdict = {
                    "FingerType.Thumb": [255, 0, 0],     # Red
                    "FingerType.Index": [0, 255, 0],     # Green
                    "FingerType.Middle": [0, 0, 255],    # Blue
                    "FingerType.Ring": [255, 255, 0],    # Yellow
                    "FingerType.Pinky": [255, 0, 255],   # Magenta
                }
                color = cdict.get(digit_type_str, [200, 200, 200])

                # The first bone (metacarpal) connects back to the wrist (index 0)
                # But to draw a continuous line strip from wrist to tip:
                strip = [[wrist.x, wrist.y, wrist.z]]

                for i, joint in enumerate(digit_points):
                    points.append([joint.x, joint.y, joint.z])
                    colors.append(color)
                    labels.append(f"{digit_type_str.split('.')[-1]}_{i}")
                    strip.append([joint.x, joint.y, joint.z])

                edges.append(strip)

            # Log points
            rr.log(f"{prefix}/joints", rr.Points3D(
                points,
                colors=colors,
                labels=labels,
                radii=5.0
            ))

            # Log edges
            rr.log(f"{prefix}/skeleton", rr.LineStrips3D(edges, colors=[[150, 150, 150]] * 5))

def main():
    parser = argparse.ArgumentParser(description="Live visualize Leap Motion tracking using Rerun.")
    parser.add_argument("--web", action="store_true", help="Serve a web viewer instead of a native window (fixes Wayland/Vulkan issues).")
    args = parser.parse_args()

    rr.init("SpikeFormerMyo_VisualizeLeap")
    if args.web:
        rr.serve_web_viewer()
    else:
        rr.spawn()

    listener = RerunTrackerListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    print("Connecting to Leap Service...")
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        print("Connected. Streaming Leap tracking to Rerun. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            print("Disconnected.")

if __name__ == "__main__":
    main()
