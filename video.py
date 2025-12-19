import cv2
import os
import shutil
import numpy as np
import torch
import gc
from dreamer import DeepDreamer
import optical_flow as flow_est

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
TEMP_DIR = "temp"

# 0.0 = Pure Warp (blurry), 1.0 = No Warp (flickering).
BLEND_WEIGHT = 0.3
UPDATE_INTERVAL = 5

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

def process_video():
    cwd = os.getcwd()
    abs_temp_dir = os.path.join(cwd, TEMP_DIR)
    
    try:
        input_frames_dir = os.path.join(abs_temp_dir, "input")
        output_frames_dir = os.path.join(abs_temp_dir, "output")
        flow_frames_dir = os.path.join(abs_temp_dir, "flow")

        print(f"Creating temporary directories at: {abs_temp_dir}")
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(flow_frames_dir, exist_ok=True)

        raft_model, raft_transforms, device = flow_est.init_raft()
        
        dreamer_args = [
            "-gpu", "mps",
            "-lap_scale", "4",
            "-image_size", "1024",
            "-save_iter", "0",
            "-print_iter", "0",
            "-num_iterations", "5",
        ]

        if not os.path.exists(INPUT_VIDEO):
            raise FileNotFoundError(f"Input video not found at: {INPUT_VIDEO}")

        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {INPUT_VIDEO}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

        frame_count = 0
        prev_frame = None
        prev_dream = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_count}/{total_frames}")

            dreamer = DeepDreamer(dreamer_args)

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
                        BLEND_WEIGHT,
                        warped_prev_dream,
                        (1 - BLEND_WEIGHT),
                        0
                    )
            else:
                cv2.imwrite(flow_path, np.zeros_like(frame))

            prev_frame = frame.copy()

            cv2.imwrite(input_frame_path, img_to_dream)
            dreamer.dream(input_frame_path, output_frame_path)

            del dreamer
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if os.path.exists(output_frame_path):
                prev_dream = cv2.imread(output_frame_path)
                if prev_dream is not None:
                    if prev_dream.shape[:2] != (height, width):
                        prev_dream = cv2.resize(prev_dream, (width, height))
                        cv2.imwrite(output_frame_path, prev_dream)
            else:
                print(f"Warning: Output missing at {output_frame_path}")

            frame_count += 1

            if frame_count % UPDATE_INTERVAL == 0:
                update_output_video(
                    OUTPUT_VIDEO, 
                    output_frames_dir, 
                    width, 
                    height, 
                    fps, 
                    frame_count
                )

        cap.release()
        
        print("Finalizing video...")
        update_output_video(OUTPUT_VIDEO, output_frames_dir, width, height, fps, frame_count)

    finally:
        if os.path.exists(abs_temp_dir):
            print(f"Cleaning up temp directory: {abs_temp_dir}")
            shutil.rmtree(abs_temp_dir)

if __name__ == "__main__":
    process_video()