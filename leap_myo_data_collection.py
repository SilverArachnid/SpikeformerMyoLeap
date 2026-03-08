import sys
import termios
import time
import traceback

import hydra
from omegaconf import DictConfig

from collection_controller import CollectionController, CollectionSettings


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
        episode_duration=float(cfg.episode_duration),
        episodes_per_session=int(cfg.max_episodes),
        save_dir=cfg.save_dir,
    )
    controller = CollectionController(settings=settings, visualize=bool(cfg.visualize))

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        controller.connect()
        controller.start_session(settings)

        new_settings = termios.tcgetattr(fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)

        print("\n==============================================")
        print(f"Ready to record up to {cfg.max_episodes} episodes of {cfg.pose_name}.")
        print(f"Target duration: {cfg.episode_duration}s per episode.")
        print("Press SPACEBAR to record the next episode.")
        print("Press 's' to stop the current episode early and save it.")
        print("Press ESC or 'q' to quit.")
        print("==============================================\n")

        while True:
            snapshot = controller.get_status_snapshot()
            if snapshot["completed_episodes"] >= snapshot["episodes_per_session"] and not snapshot["recording"]:
                print("Reached configured episode count. Ending session.")
                break

            if is_key_pressed():
                char = sys.stdin.read(1)

                if char in ("q", "\x1b"):
                    print("\nQuitting...")
                    break
                if char == " " and not snapshot["recording"]:
                    print(f"\n[Episode {snapshot['completed_episodes'] + 1}/{snapshot['episodes_per_session']}] Recording started...")
                    controller.start_episode()
                elif char.lower() == "s" and snapshot["recording"]:
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
