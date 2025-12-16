import cv2
import os
import shutil
import numpy as np
from dreamer import DeepDreamer
import optical_flow as flow_est

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
TEMP_DIR = "temp"

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
            "-save_iter", "0",
            "-print_iter", "0",
            "-num_iterations", "1",
        ]
        dreamer = DeepDreamer(dreamer_args)
        print("Dreamer model loaded.")

        if not os.path.exists(INPUT_VIDEO):
            raise FileNotFoundError(f"Input video not found at: {os.path.join(cwd, INPUT_VIDEO)}")

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

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_count == 0:
                    print("Error: Video opened but no frames could be read.")
                break

            print(f"Processing frame {frame_count}/{total_frames}")

            input_frame_path = os.path.join(input_frames_dir, f"frame_{frame_count:06d}.jpg")
            output_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:06d}.jpg")
            flow_path = os.path.join(flow_frames_dir, f"flow_{frame_count:06d}.jpg")

            success = cv2.imwrite(input_frame_path, frame)
            if not success:
                raise IOError(f"Failed to write input frame to {input_frame_path}")

            if prev_frame is not None:
                flow_vis = flow_est.estimate_flow(
                    prev_frame, 
                    frame, 
                    raft_model, 
                    raft_transforms, 
                    device
                )
                success = cv2.imwrite(flow_path, flow_vis)
                if not success:
                     print(f"Warning: Failed to write flow frame {flow_path}")
            else:
                cv2.imwrite(flow_path, np.zeros_like(frame))

            prev_frame = frame.copy()

            dreamer.dream(input_frame_path, output_frame_path)
            
            if not os.path.exists(output_frame_path):
                 print(f"Warning: Dreamer did not produce output at {output_frame_path}")

            frame_count += 1
            if frame_count > 100: 
                print("Frame limit reached for testing.")
                break

        cap.release()
        print(f"Processed {frame_count} frames.")

        print("Reassembling video...")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

        frames_written = 0
        for i in range(frame_count):
            output_frame = os.path.join(output_frames_dir, f"frame_{i:06d}.jpg")
            
            if not os.path.exists(output_frame):
                print(f"Warning: Missing frame {output_frame}")
                continue
                
            frame_read = cv2.imread(output_frame)
            if frame_read is None:
                print(f"Warning: Could not read frame {output_frame}")
                continue

            if frame_read.shape[:2] != (height, width):
                frame_read = cv2.resize(frame_read, (width, height))

            out.write(frame_read)
            frames_written += 1

        out.release()
        print(f"Video saved to {OUTPUT_VIDEO} with {frames_written} frames.")

    finally:
        if os.path.exists(abs_temp_dir):
            print(f"Cleaning up temp directory: {abs_temp_dir}")
            shutil.rmtree(abs_temp_dir)

if __name__ == "__main__":
    process_video()