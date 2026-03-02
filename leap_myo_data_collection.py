import os
import sys
import time
import termios
import threading
import uuid
import json
import traceback

import hydra
from omegaconf import DictConfig
import numpy as np
import pandas as pd
import rerun as rr

import pyomyo
import leap

def is_key_pressed():
    """Non-blocking keyboard check for Linux."""
    import select
    dr,dw,de = select.select([sys.stdin], [], [], 0)
    return dr != []

class LeapRerunListener(leap.Listener):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pose_data = [] # stores tuples: (timestamp_ms, x1,y1,z1...x21,y21,z21)
        self.pose_lock = threading.Lock()
        
        # Shared recording state set by main thread
        self.is_recording = False
        self.episode_start_time = None

    def on_tracking_event(self, event):
        # We assume one primary hand for SpikeFormerMyo dataset
        if not event.hands:
            return
            
        hand = event.hands[0]
        
        # Extract joints exactly like visualize_leap.py
        wrist = hand.arm.next_joint
        joints_xyz = [wrist.x, wrist.y, wrist.z]
        
        # Digits: order is Thumb, Index, Middle, Ring, Pinky
        for digit in hand.digits:
            joints_xyz.extend([
                digit.metacarpal.next_joint.x, digit.metacarpal.next_joint.y, digit.metacarpal.next_joint.z,
                digit.proximal.next_joint.x, digit.proximal.next_joint.y, digit.proximal.next_joint.z,
                digit.intermediate.next_joint.x, digit.intermediate.next_joint.y, digit.intermediate.next_joint.z,
                digit.distal.next_joint.x, digit.distal.next_joint.y, digit.distal.next_joint.z
            ])

        # If visualize=True, log skeleton to Rerun
        if self.cfg.visualize:
            hand_type = "Left" if str(hand.type) == "HandType.Left" else "Right"
            prefix = f"hand/{hand_type}"
            points = np.array(joints_xyz).reshape(21, 3)
            
            # Reconstruct strips for visualization
            edges = []
            for i in range(5):
                start_idx = 1 + (i * 4)
                strip = [points[0]] + list(points[start_idx : start_idx + 4])
                edges.append(strip)
            
            rr.log(f"{prefix}/joints", rr.Points3D(points, radii=5.0))
            rr.log(f"{prefix}/skeleton", rr.LineStrips3D(edges))

        # Only record if state says so
        if self.is_recording and self.episode_start_time is not None:
            timestamp = (time.perf_counter() - self.episode_start_time) * 1000.0  # ms
            with self.pose_lock:
                self.pose_data.append((timestamp, *joints_xyz))


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Setup optional Rerun visualization
    if cfg.visualize:
        rr.init("SpikeFormerMyo_DataCollection")
        if getattr(cfg, "web_viewer", False):
            rr.serve_web_viewer()
        else:
            rr.spawn()

    # Output directory
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # ---------------------- EMG Setup ----------------------
    print("Connecting to Myo...")
    m = pyomyo.Myo(mode=pyomyo.emg_mode.PREPROCESSED)
    m.connect()
    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)
    
    myo_data = [] # stores (timestamp_ms, ch1...ch8)
    episode_start_time = None
    is_recording = False

    def myo_handler(emg, movement):
        if cfg.visualize:
            for i, val in enumerate(emg):
                rr.log(f"emg/ch{i+1}", rr.TimeSeriesScalar(val))
                
        if is_recording and episode_start_time is not None:
            timestamp = (time.perf_counter() - episode_start_time) * 1000.0
            myo_data.append((timestamp, *emg))

    m.add_emg_handler(myo_handler)

    # ---------------------- Leap Setup ----------------------
    print("Connecting to Leap Motion...")
    leap_listener = LeapRerunListener(cfg)
    connection = leap.Connection()
    connection.add_listener(leap_listener)

    # Set up terminal for non-blocking key presses
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        connection.open()
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        # Update terminal config to read raw keystrokes
        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)

        print("\n==============================================")
        print(f"Ready to record {cfg.max_episodes} episodes of {cfg.pose_name}.")
        print(f"Target duration: {cfg.episode_duration}s per episode.")
        print("Press SPACEBAR to trigger a recording. Press ESC or 'q' to quit.")
        print("==============================================\n")

        current_episode = 0

        while current_episode < cfg.max_episodes:
            if is_key_pressed():
                char = sys.stdin.read(1)
                
                if char == 'q' or char == '\x1b': # ESC or q
                    print("\nQuitting early...")
                    break
                elif char == ' ':
                    if not is_recording:
                        print(f"\n[Episode {current_episode + 1}/{cfg.max_episodes}] Recording started...")
                        
                        myo_data.clear()
                        with leap_listener.pose_lock:
                            leap_listener.pose_data.clear()
                            
                        episode_start_time = time.perf_counter()
                        is_recording = True
                        leap_listener.episode_start_time = episode_start_time
                        leap_listener.is_recording = True
                        
                        # Busy-wait block for the duration of the episode to keep pumping Myo
                        while time.perf_counter() - episode_start_time < cfg.episode_duration:
                            m.run()
                            
                        # Stop recording
                        is_recording = False
                        leap_listener.is_recording = False
                        
                        m.vibrate(1)
                        print("Recording stopped. Saving...")
                        
                        # Process and Save
                        episode_id = str(uuid.uuid4())
                        ep_folder = os.path.join(cfg.save_dir, f"{cfg.pose_name}_ep{current_episode + 1}")
                        os.makedirs(ep_folder, exist_ok=True)
                        
                        # Save EMG
                        if myo_data:
                            ts_emg, *channels = zip(*myo_data)
                            emg_arr = np.array(channels).T
                            emg_df = pd.DataFrame(emg_arr, columns=[f"Channel_{i+1}" for i in range(8)])
                            emg_df.insert(0, "Timestamp_ms", ts_emg)
                            emg_path = os.path.join(ep_folder, "emg.csv")
                            emg_df.to_csv(emg_path, index=False)
                            print(f"   -> Saved EMG: {len(emg_df)} samples, max length {emg_df['Timestamp_ms'].iloc[-1]:.1f}ms")
                        else:
                            print("   -> ERROR: No EMG data collected.")

                        # Save Pose
                        with leap_listener.pose_lock:
                            if leap_listener.pose_data:
                                ts_pose, *pose_pts = zip(*leap_listener.pose_data)
                                pose_arr = np.array(pose_pts).T
                                
                                # Column Names matching SpikeformerMyo
                                landmark_names = [
                                    'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                                    'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                                    'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
                                    'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                                    'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
                                ]
                                
                                columns = []
                                for name in landmark_names:
                                    columns.extend([f"{name}_X", f"{name}_Y", f"{name}_Z"])
                                    
                                pose_df = pd.DataFrame(pose_arr, columns=columns)
                                pose_df.insert(0, "Timestamp_ms", ts_pose)
                                pose_path = os.path.join(ep_folder, "pose.csv")
                                pose_df.to_csv(pose_path, index=False)
                                print(f"   -> Saved Pose: {len(pose_df)} frames, max length {pose_df['Timestamp_ms'].iloc[-1]:.1f}ms")
                            else:
                                print("   -> ERROR: No Leap Pose data collected.")
                                
                        # Save Meta
                        metadata = {
                            "episode_id": episode_id,
                            "pose_name": cfg.pose_name,
                            "subject_id": cfg.subject_id,
                            "episode_number": current_episode + 1,
                            "duration_seconds": cfg.episode_duration,
                            "sample_count_emg": len(myo_data),
                            "sample_count_pose": len(leap_listener.pose_data)
                        }
                        with open(os.path.join(ep_folder, "meta.json"), "w") as f:
                            json.dump(metadata, f, indent=4)
                            
                        current_episode += 1
                        
            # Keep Myo alive while waiting for input
            m.run()

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        traceback.print_exc()
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        m.disconnect()
        print(f"\nDone. Collected {current_episode} episodes.")

if __name__ == "__main__":
    main()
