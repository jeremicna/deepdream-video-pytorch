import cv2
import os
import shutil
import numpy as np
import torch
import argparse
import contextlib
from dreamer import DeepDreamer
import optical_flow as flow_est

def get_suppressor(show_output):
    if show_output:
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

def calculate_occlusion_mask(current_frame, warped_prev_frame, threshold=30):
    height, width = current_frame.shape[:2]
    
    base_size = int(np.sqrt(height * width))
    
    morph_size = max(3, int(base_size * 0.003))
    morph_size = morph_size if morph_size % 2 == 1 else morph_size + 1
    
    blur_size = max(5, int(base_size * 0.009))
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    
    diff = cv2.absdiff(current_frame, warped_prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((morph_size, morph_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    float_mask = mask.astype(np.float32) / 255.0
    
    return np.expand_dims(float_mask, axis=2), mask

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
    
    output_suppressor = get_suppressor(args.verbose)

    keys_to_remove = ["-content_image", "-output_image"]
    clean_dreamer_args = filter_args(dreamer_args, keys_to_remove)

    try:
        input_frames_dir = os.path.join(abs_temp_dir, "input")
        output_frames_dir = os.path.join(abs_temp_dir, "output")
        flow_frames_dir = os.path.join(abs_temp_dir, "flow")
        mask_frames_dir = os.path.join(abs_temp_dir, "mask")

        print(f"Creating temporary directories at: {abs_temp_dir}")
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(flow_frames_dir, exist_ok=True)
        os.makedirs(mask_frames_dir, exist_ok=True)

        if not args.independent:
            print("Initializing Optical Flow (RAFT)...")
            with output_suppressor:
                raft_model, raft_transforms, device = flow_est.init_raft()
        else:
            print("Mode: Independent (Temporal consistency disabled)")
        
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

            with output_suppressor:
                dreamer = DeepDreamer(clean_dreamer_args)

            input_frame_path = os.path.join(input_frames_dir, f"frame_{frame_count:06d}.jpg")
            output_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:06d}.jpg")
            flow_path = os.path.join(flow_frames_dir, f"flow_{frame_count:06d}.jpg")
            mask_path = os.path.join(mask_frames_dir, f"mask_{frame_count:06d}.jpg")

            img_to_dream = frame.copy()

            if not args.independent:
                if prev_frame is not None:
                    with output_suppressor:
                        flow_data, flow_vis = flow_est.estimate_flow(
                            prev_frame, frame, raft_model, raft_transforms, device
                        )
                    cv2.imwrite(flow_path, flow_vis)

                    if prev_dream is not None:
                        if prev_dream.shape[:2] != frame.shape[:2]:
                            prev_dream = cv2.resize(prev_dream, (frame.shape[1], frame.shape[0]))

                        warped_prev_dream = flow_est.warp_image(prev_dream, flow_data)
                        warped_prev_frame = flow_est.warp_image(prev_frame, flow_data)

                        mask, mask_vis = calculate_occlusion_mask(frame, warped_prev_frame, threshold=30)
                        cv2.imwrite(mask_path, mask_vis)

                        guided_dream = (mask * warped_prev_dream) + ((1 - mask) * frame)
                        guided_dream = guided_dream.astype(np.uint8)

                        img_to_dream = cv2.addWeighted(
                            frame, args.blend, guided_dream, (1 - args.blend), 0
                        )
                else:
                    cv2.imwrite(flow_path, np.zeros_like(frame))
                    cv2.imwrite(mask_path, 255 * np.ones((height, width), dtype=np.uint8))

            prev_frame = frame.copy()
            cv2.imwrite(input_frame_path, img_to_dream)

            with output_suppressor:
                dreamer.dream(input_frame_path, output_frame_path)
            
            del dreamer
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            if os.path.exists(output_frame_path):
                prev_dream = cv2.imread(output_frame_path)
                if prev_dream is not None:
                    if output_width is None:
                        output_height, output_width = prev_dream.shape[:2]
            else:
                print(f"Warning: Output missing at {output_frame_path}")

            frame_count += 1
            if frame_count % args.update_interval == 0 and output_width is not None:
                update_output_video(args.output_video, output_frames_dir, output_width, output_height, fps, frame_count)

        cap.release()
        if output_width is not None:
            update_output_video(args.output_video, output_frames_dir, output_width, output_height, fps, frame_count)

    finally:
        if os.path.exists(abs_temp_dir) and not args.keep_temp:
            print(f"Cleaning up: {abs_temp_dir}")
            shutil.rmtree(abs_temp_dir)

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