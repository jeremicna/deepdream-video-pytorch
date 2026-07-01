import cv2
import os
import shutil
import torch
import argparse
import contextlib

from dreamer import DeepDreamer
import optical_flow as flow_est


@contextlib.contextmanager
def suppress_output(show_output):
    if show_output:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def filter_args(args_list, keys_to_remove):
    filtered = []
    skip_next = False
    for arg in args_list:
        if skip_next:
            skip_next = False
            continue
        if arg in keys_to_remove:
            skip_next = True
            continue
        if any(arg.startswith(f"{key}=") for key in keys_to_remove):
            continue
        filtered.append(arg)
    return filtered


def create_workspace(temp_dir):
    root = os.path.abspath(temp_dir)
    workspace = {
        "root": root,
        "input": os.path.join(root, "input"),
        "output": os.path.join(root, "output"),
        "flow": os.path.join(root, "flow"),
        "mask": os.path.join(root, "mask"),
    }
    print(f"Creating temporary directories at: {workspace['root']}")
    for path in (workspace["input"], workspace["output"], workspace["flow"], workspace["mask"]):
        os.makedirs(path, exist_ok=True)
    return workspace


def frame_paths(workspace, frame_number):
    return {
        "dream_input": os.path.join(workspace["input"], f"frame_{frame_number:06d}.jpg"),
        "dream_output": os.path.join(workspace["output"], f"frame_{frame_number:06d}.jpg"),
        "flow": os.path.join(workspace["flow"], f"flow_{frame_number:06d}.jpg"),
        "mask": os.path.join(workspace["mask"], f"mask_{frame_number:06d}.jpg"),
    }


def cleanup_workspace(workspace):
    if os.path.exists(workspace["root"]):
        print(f"Cleaning up: {workspace['root']}")
        shutil.rmtree(workspace["root"])


def clear_accelerator_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_video_args(args):
    if not 0.0 <= args.blend <= 1.0:
        raise ValueError("-blend must be between 0.0 and 1.0")
    if args.update_interval < 0:
        raise ValueError("-update_interval must be 0 or greater")


def create_video_writer(output_path, fps, width, height):
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise ValueError(f"Could not create output video: {output_path}")


def update_output_video(output_path, frames_dir, width, height, fps, count):
    temp_output = output_path + ".tmp.mp4"
    out, codec = create_video_writer(temp_output, fps, width, height)

    frames_written = 0
    for i in range(count):
        frame_path = os.path.join(frames_dir, f"frame_{i:06d}.jpg")
        if os.path.exists(frame_path):
            frame_read = cv2.imread(frame_path)
            if frame_read is not None:
                if frame_read.shape[:2] != (height, width):
                    frame_read = cv2.resize(frame_read, (width, height))
                out.write(frame_read)
                frames_written += 1

    out.release()
    os.replace(temp_output, output_path)

    print(f"[Video Update] Refreshed {output_path} with {frames_written} frames.")
    print(f"[Video Update] Codec: {codec}")
    if codec == "mp4v":
        print("[Video Update] Note: mp4v is valid MP4, but VS Code's preview may not play it.")


def load_temporal_guidance(verbose):
    print("Initializing Optical Flow (RAFT)...")
    with suppress_output(verbose):
        return flow_est.TemporalGuidance.load_raft()


def process_video(args, dreamer_args):
    validate_video_args(args)

    keys_to_remove = ["-content_image", "-output_image"]
    clean_dreamer_args = filter_args(dreamer_args, keys_to_remove)

    if not os.path.exists(args.content_video):
        raise FileNotFoundError(f"Input video not found at: {args.content_video}")

    workspace = create_workspace(args.temp_dir)

    try:
        temporal_guidance = None
        if args.independent:
            print("Mode: Independent (Temporal consistency disabled)")
        else:
            temporal_guidance = load_temporal_guidance(args.verbose)

        output_width = None
        output_height = None
        frame_count = 0
        cap = cv2.VideoCapture(args.content_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {args.content_video}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

            prev_frame = None
            prev_dream = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                print(f"Processing frame {frame_count}/{total_frames}")

                with suppress_output(args.verbose):
                    dreamer = DeepDreamer(clean_dreamer_args)

                paths = frame_paths(workspace, frame_count)
                img_to_dream = frame.copy()

                if temporal_guidance is not None:
                    with suppress_output(args.verbose):
                        img_to_dream, flow_vis, mask_vis = temporal_guidance.guide(
                            frame, prev_frame, prev_dream, blend=args.blend
                        )
                    cv2.imwrite(paths["flow"], flow_vis)
                    cv2.imwrite(paths["mask"], mask_vis)

                prev_frame = frame.copy()
                cv2.imwrite(paths["dream_input"], img_to_dream)

                with suppress_output(args.verbose):
                    dreamer.dream(paths["dream_input"], paths["dream_output"])

                del dreamer
                clear_accelerator_cache()

                if os.path.exists(paths["dream_output"]):
                    prev_dream = cv2.imread(paths["dream_output"])
                    if prev_dream is not None and output_width is None:
                        output_height, output_width = prev_dream.shape[:2]
                    elif prev_dream is None:
                        print(f"Warning: Could not read output frame at {paths['dream_output']}")
                else:
                    print(f"Warning: Output missing at {paths['dream_output']}")

                frame_count += 1
                if (
                    args.update_interval > 0
                    and frame_count % args.update_interval == 0
                    and output_width is not None
                ):
                    update_output_video(
                        args.output_video,
                        workspace["output"],
                        output_width,
                        output_height,
                        fps,
                        frame_count,
                    )

        finally:
            cap.release()

        if output_width is not None:
            update_output_video(
                args.output_video,
                workspace["output"],
                output_width,
                output_height,
                fps,
                frame_count,
            )

    finally:
        if not args.keep_temp:
            cleanup_workspace(workspace)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video DeepDream CLI")
    parser.add_argument("-content_video", type=str, default="input.mp4", help="Path to input video")
    parser.add_argument("-output_video", type=str, default="output.mp4", help="Path to output video")
    parser.add_argument("-temp_dir", type=str, default="temp", help="Directory for temporary frames")
    parser.add_argument("-blend", type=float, default=0.5, help="Blend weight")
    parser.add_argument("-update_interval", type=int, default=5, help="Update output video every N frames")
    
    parser.add_argument("-verbose", action="store_true", help="Enable detailed logs")
    parser.add_argument("-independent", action="store_true", help="Disable temporal consistency")
    
    parser.add_argument("-keep_temp", action="store_true", help="Do not delete temp directory")

    args, unknown_args = parser.parse_known_args()
    process_video(args, unknown_args)
