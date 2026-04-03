import sys
import termios
import time
import traceback

import hydra
from omegaconf import DictConfig

from spikeformer_myo_leap.collection.controller import CollectionController
from spikeformer_myo_leap.data.contracts import CollectionSettings


def is_key_pressed():
    import select

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    return readable != []


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    settings = CollectionSettings(
        subject_id=cfg.subject_id,
        session_name=cfg.session_name,
        pose_name=cfg.pose_name,
        recording_mode=str(cfg.recording_mode),
        episode_duration=float(cfg.episode_duration),
        episodes_per_session=int(cfg.max_episodes),
        save_dir=cfg.save_dir,
    )
    controller = CollectionController(settings=settings, visualize=bool(cfg.visualize))

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    last_status_message = ""
    last_aborted_episode = ""

    try:
        controller.connect()
        controller.start_session(settings)

        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)

        print("\n==============================================")
        print(f"Ready to record up to {cfg.max_episodes} episodes of {cfg.pose_name}.")
        print(f"Target duration: {cfg.episode_duration}s per episode.")
        if settings.recording_mode == "continuous":
            print("Press SPACEBAR to record the remaining episode slots as one continuous block.")
            print("Press 's' to stop the current block early and save only completed full segments.")
        else:
            print("Press SPACEBAR to record the next episode.")
            print("Press 's' to stop the current episode early and save it.")
        print("Press ESC or 'q' to quit.")
        print("==============================================\n")

        while True:
            snapshot = controller.get_status_snapshot()
            if snapshot["status_message"] != last_status_message:
                print(f"[Status] {snapshot['status_message']}")
                last_status_message = snapshot["status_message"]
            if snapshot["last_aborted_episode"] and snapshot["last_aborted_episode"] != last_aborted_episode:
                print(
                    f"[Recovery] {snapshot['last_aborted_episode']} was aborted. "
                    "Waiting for healthy sensor streams before the next recording."
                )
                last_aborted_episode = snapshot["last_aborted_episode"]
            if snapshot["completed_episodes"] >= snapshot["episodes_per_session"] and not snapshot["recording"]:
                print("Reached configured episode count. Ending session.")
                break

            if is_key_pressed():
                char = sys.stdin.read(1)

                if char in ("q", "\x1b"):
                    print("\nQuitting...")
                    break
                if char == " " and not snapshot["recording"]:
                    if snapshot.get("finalizing_episode"):
                        print("\n[Status] Previous episode is still being saved. Please wait.")
                        continue
                    if not snapshot.get("myo_connected") or not snapshot.get("leap_connected"):
                        print("\n[Status] Cannot record yet: both sensors must be connected.")
                        continue
                    if not snapshot.get("myo_streaming") or not snapshot.get("leap_streaming"):
                        waiting = []
                        if not snapshot.get("myo_streaming"):
                            waiting.append("Myo")
                        if not snapshot.get("leap_streaming"):
                            waiting.append("Leap")
                        print(f"\n[Status] Cannot record yet: waiting for healthy sensor data: {', '.join(waiting)}")
                        continue
                    try:
                        if settings.recording_mode == "continuous":
                            remaining = snapshot["episodes_per_session"] - snapshot["completed_episodes"]
                            print(
                                f"\n[Continuous Block {snapshot['completed_episodes'] + 1}/"
                                f"{snapshot['episodes_per_session']}] Recording started "
                                f"for {remaining} episode window(s)..."
                            )
                        else:
                            print(
                                f"\n[Episode {snapshot['completed_episodes'] + 1}/"
                                f"{snapshot['episodes_per_session']}] Recording started..."
                            )
                        controller.start_episode()
                    except RuntimeError as exc:
                        print(f"\n[Status] Could not start recording: {exc}")
                elif char.lower() == "s" and snapshot["recording"]:
                    if settings.recording_mode == "continuous":
                        print("\nStopping current continuous block early...")
                    else:
                        print("\nStopping current episode early...")
                    controller.stop_episode()

            time.sleep(0.02)

    except Exception as exc:
        print(f"\nAn error occurred: {exc}")
        traceback.print_exc()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        controller.close()
        snapshot = controller.get_status_snapshot()
        print(f"\nDone. Collected {snapshot['completed_episodes']} episodes.")


if __name__ == "__main__":
    main()
