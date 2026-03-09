import os

import rerun as rr


def prepare_native_viewer_environment() -> list[str]:
    notes: list[str] = []

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    has_x11 = bool(os.environ.get("DISPLAY"))

    if (session_type == "wayland" or has_wayland) and has_x11 and "WINIT_UNIX_BACKEND" not in os.environ:
        os.environ["WINIT_UNIX_BACKEND"] = "x11"
        notes.append("WINIT_UNIX_BACKEND=x11")

    if "WGPU_BACKEND" not in os.environ:
        os.environ["WGPU_BACKEND"] = "gl"
        notes.append("WGPU_BACKEND=gl")

    if "LIBGL_ALWAYS_SOFTWARE" not in os.environ:
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        notes.append("LIBGL_ALWAYS_SOFTWARE=1")

    return notes


def should_use_web_viewer(force_web: bool = False, force_native: bool = False) -> bool:
    if force_web:
        return True
    if force_native:
        return False

    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    has_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    has_x11 = bool(os.environ.get("DISPLAY"))
    if not has_x11 and not has_wayland:
        return True
    return False


def init_rerun(app_id: str, force_web: bool = False, force_native: bool = False) -> bool:
    rr.init(app_id)
    use_web = should_use_web_viewer(force_web=force_web, force_native=force_native)
    if use_web:
        grpc_uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=grpc_uri, open_browser=True)
        print("Rerun web viewer URL: http://127.0.0.1:9090")
    else:
        notes = prepare_native_viewer_environment()
        if notes:
            print(f"Native Rerun viewer environment: {', '.join(notes)}")
        rr.spawn()
    return use_web
