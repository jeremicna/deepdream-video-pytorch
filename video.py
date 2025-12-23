import cv2
import os
import shutil
import numpy as np
import torch
import argparse
import sys
import contextlib
from dreamer import DeepDreamer
import optical_flow as flow_est

def get_suppressor(debug_mode):
    if debug_mode:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(open(os.devnull, 'w'))

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
        filtered.append(arg)
    return filtered

def update_output_video(output_path, frames_dir, width, height, fps, count):
    temp_output = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
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
    
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_output, output_path)
    
    print(f"[Video Update] Refreshed {output_path} with {frames_written} frames.")

def process_video(args, dreamer_args):
    cwd = os.getcwd()
    abs_temp_dir = os.path.join(cwd, args.temp_dir)
    suppressor = get_suppressor(args.debug)

    keys_to_remove = ["-content_image", "-output_image"]
    clean_dreamer_args = filter_args(dreamer_args, keys_to_remove)

    try:
        input_frames_dir = os.path.join(abs_temp_dir, "input")
        output_frames_dir = os.path.join(abs_temp_dir, "output")
        flow_frames_dir = os.path.join(abs_temp_dir, "flow")

        print(f"Creating temporary directories at: {abs_temp_dir}")
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(flow_frames_dir, exist_ok=True)

        print("Initializing Optical Flow (RAFT)...")
        with suppressor:
            raft_model, raft_transforms, device = flow_est.init_raft()
        
        if args.debug:
            print(f"Forwarding arguments to DeepDreamer: {clean_dreamer_args}")

        if not os.path.exists(args.content_video):
            raise FileNotFoundError(f"Input video not found at: {args.content_video}")

        cap = cv2.VideoCapture(args.content_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {args.content_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

        output_width = None
        output_height = None

        frame_count = 0
        prev_frame = None
        prev_dream = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_count}/{total_frames}")

            with suppressor:
                dreamer = DeepDreamer(clean_dreamer_args)

            input_frame_path = os.path.join(
                input_frames_dir, f"frame_{frame_count:06d}.jpg"
            )
            output_frame_path = os.path.join(
                output_frames_dir, f"frame_{frame_count:06d}.jpg"
            )
            flow_path = os.path.join(
                flow_frames_dir, f"flow_{frame_count:06d}.jpg"
            )

            img_to_dream = frame.copy()

            if prev_frame is not None:
                with suppressor:
                    flow_data, flow_vis = flow_est.estimate_flow(
                        prev_frame,
                        frame,
                        raft_model,
                        raft_transforms,
                        device
                    )

                cv2.imwrite(flow_path, flow_vis)

                if prev_dream is not None:
                    if prev_dream.shape[:2] != frame.shape[:2]:
                        prev_dream = cv2.resize(
                            prev_dream,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )

                    warped_prev_dream = flow_est.warp_image(
                        prev_dream, flow_data
                    )

                    img_to_dream = cv2.addWeighted(
                        frame,
                        args.blend,
                        warped_prev_dream,
                        (1 - args.blend),
                        0
                    )
            else:
                cv2.imwrite(flow_path, np.zeros_like(frame))

            prev_frame = frame.copy()

            cv2.imwrite(input_frame_path, img_to_dream)

            with suppressor:
                dreamer.dream(input_frame_path, output_frame_path)

            del dreamer

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if os.path.exists(output_frame_path):
                prev_dream = cv2.imread(output_frame_path)
                if prev_dream is not None:
                    if output_width is None or output_height is None:
                        output_height, output_width = prev_dream.shape[:2]
                        print(f"DeepDream output size detected: {output_width}x{output_height}")
                    
            else:
                print(f"Warning: Output missing at {output_frame_path}")

            frame_count += 1

            if frame_count % args.update_interval == 0 and output_width is not None:
                update_output_video(
                    args.output_video, 
                    output_frames_dir, 
                    output_width, 
                    output_height, 
                    fps, 
                    frame_count
                )

        cap.release()
        
        if output_width is not None:
            print("Finalizing video...")
            update_output_video(args.output_video, output_frames_dir, output_width, output_height, fps, frame_count)
        else:
            print("Error: No frames were successfully processed")

    finally:
        if os.path.exists(abs_temp_dir) and not args.keep_temp:
            print(f"Cleaning up temp directory: {abs_temp_dir}")
            shutil.rmtree(abs_temp_dir)
        elif args.keep_temp:
             print(f"Keeping temp directory: {abs_temp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video DeepDream CLI with Optical Flow Stability")
    
    parser.add_argument("-content_video", type=str, default="input.mp4", help="Path to input video")
    parser.add_argument("-output_video", type=str, default="output.mp4", help="Path to output video")
    
    parser.add_argument("-temp_dir", type=str, default="temp", help="Directory for temporary frames")
    parser.add_argument("-blend", type=float, default=0.5, help="Blend weight (0.0=Pure Warp, 1.0=No Warp)")
    parser.add_argument("-update_interval", type=int, default=5, help="Update output video every N frames")
    parser.add_argument("-debug", action="store_true", help="Enable stdout from the dreamer")
    parser.add_argument("-keep_temp", action="store_true", help="Do not delete temp directory after finishing")

    args, unknown_args = parser.parse_known_args()

    process_video(args, unknown_args)