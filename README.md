# deepdream-video-pytorch

This is a fork of [neural-dream](https://github.com/ProGamerGov/neural-dream), a PyTorch implementation of DeepDream. This fork introduces **optical flow estimation** and **occlusion masking** to apply DeepDream to videos with temporal consistency.

## Features

* **Standard DeepDream**: The original single-image implementation.
* **Video DeepDream**: New CLI (`video_dream.py`) that uses RAFT Optical Flow to warp previous dream frames into the current frame, ensuring smooth transitions and object tracking.
* **Occlusion Masking**: Automatically detects when objects move in front of one another to prevent "ghosting" artifacts.

## Demo

### With temporal consistency

https://github.com/user-attachments/assets/0f3302a1-d670-45dd-afb6-5a99b561e0f6

https://github.com/user-attachments/assets/e7630c8c-3ed9-4b42-bba9-a666338a1784

### With frames processed independently

### Inputs

## Setup

### Dependencies

This project requires the following key packages:

* **PyTorch**
* **torchvision**
* **OpenCV**
* **NumPy**
* **Pillow**

**Install Dependencies**:

```bash
pip install -r requirements.txt
```

**Download Models**:
Run the download script to fetch the standard Inception/GoogLeNet models:

```bash
python models/download_models.py
```

To download all compatible models:

```bash
python models/download_models.py -models all-caffe-googlenet
```

## Usage

### 1. Video DeepDream

To dream on a video, use the `video_dream.py` script. This wrapper accepts specific video arguments **and** any argument accepted by the standard image dreamer (e.g., layers, octaves, iterations).

**Basic Video Command:**

```bash
python video_dream.py -content_video input.mp4 -output_video output.mp4 -num_iterations 1
```

**Note:** For video processing, **we recommend using `-num_iterations 1`**. The temporal consistency from optical flow means each frame builds on the previous dream, so fewer iterations per frame are needed compared to single images.

**Video-Specific Arguments:**

| Argument | Default | Description |
| --- | --- | --- |
| `-content_video` | `input.mp4` | Path to the source video file. |
| `-output_video` | `output.mp4` | Path where the final video will be saved. |
| `-blend` | `0.5` | **(0.0 - 1.0)**: Mix ratio between the raw video frame and the warped previous dream. Higher values (closer to 1.0) use more of the raw frame; lower values (closer to 0.0) preserve more of the previous hallucinations. |
| `-independent` | `False` | **Flag**: If set, disables temporal consistency (Optical Flow). Every frame is dreamed on independently (causes flickering). |
| `-update_interval` | `5` | Updates the output video file on disk every N frames (allows you to preview progress while running). |
| `-temp_dir` | `temp` | Directory to store extracted frames, flow data, and masks during processing. |
| `-keep_temp` | `False` | **Flag**: If set, the temporary directory is not deleted after processing finishes. |
| `-verbose` | `False` | **Flag**: Enable detailed logs (prints DeepDream iteration logs for every frame). |

---

### 2. Standard DeepDream Arguments

All of the following arguments are from the single frame implementation, and you can mix and match any of these with the video-specific arguments above. Refer to [neural-dream](https://github.com/ProGamerGov/neural-dream) for more information on single frame parameters.

**Example combining video and standard args:**

```bash
python video_dream.py -content_video test.mp4 -dream_layers inception_4d -num_iterations 1 -octave_scale 0.7 -image_size 512
```

**For single image processing only:**

```bash
python neural_dream.py -content_image <image.jpg> -dream_layers inception_4c -num_iterations 10
```

**Note**: Paths to images should not contain the `~` character; use relative or absolute paths.

#### Standard Options

* `-image_size`: Maximum side length (in pixels) of the generated image. Default is 512.
* `-gpu`: Zero-indexed ID of the GPU to use; for CPU mode set `-gpu` to `c`; for MPS mode (Apple Silicon) set `-gpu` to `mps`.

#### Optimization Options

* `-dream_weight`: How much to weight DeepDream. Default is `1e3`.
* `-tv_weight`: Weight of total-variation (TV) regularization; helps smooth the image. Default `0`.
* `-l2_weight`: Weight of latent state regularization. Default `0`.
* `-num_iterations`: Number of iterations. Default is `10`. **For video, use `1`** (temporal consistency reduces the need for multiple iterations per frame).
* `-init`: Initialization method: `image` (content image) or `random` (noise). Default `image`.
* `-jitter`: Apply jitter to image. Default `32`.
* `-layer_sigma`: Apply gaussian blur to image. Default `0` (disabled).
* `-optimizer`: `lbfgs` or `adam`. Default `adam`.
* `-learning_rate`: Learning rate (step size). Default `1.5`.
* `-normalize_weights`: Divide dream weights by the number of channels.
* `-loss_mode`: Loss mode: `bce`, `mse`, `mean`, `norm`, or `l2`. Default `l2`.

#### Output Options

* `-output_image`: Name of the output image. Default `out.png`.
* `-output_start_num`: Number to start output image names at. Default `1`.
* `-print_iter`: Print progress every N iterations.
* `-save_iter`: Save image every N iterations.

#### Layer & Channel Options

* `-dream_layers`: Comma-separated list of layer names to use.
* `-channels`: Comma-separated list of channels to use.
* `-channel_mode`: Selection mode: `all`, `strong`, `avg`, `weak`, or `ignore`.
* `-channel_capture`: `once` or `octave_iter`.

#### Octave Options

* `-num_octaves`: Number of octaves per iteration. Default `4`.
* `-octave_scale`: Resize value. Default `0.6`.
* `-octave_iter`: Iterations (steps) per octave. Default `50`.
* `-octave_mode`: `normal`, `advanced`, `manual_max`, `manual_min`, or `manual`.

#### Laplacian Pyramid Options

* `-lap_scale`: Number of layers in laplacian pyramid. Default `0` (disabled).
* `-sigma`: Strength of gaussian blur in pyramids. Default `1`.

#### Zoom & Tile Options

* `-zoom`: Amount to zoom in.
* `-zoom_mode`: `percentage` or `pixel`.
* `-tile_size`: Desired tile size. Default `0` (disabled).
* `-overlap_percent`: Percentage of overlap for tiles. Default `50`.

#### Other Options

* `-original_colors`: Set to `1` to keep content image colors.
* `-model_file`: Path to `.pth` file. Default is VGG-19.
* `-model_type`: `caffe`, `pytorch`, `keras`, or `auto`.
* `-backend`: `nn`, `cudnn`, `openmp`, or `mkl`.
* `-cudnn_autotune`: Use built-in cuDNN autotuner (slower start, faster run).

## Frequently Asked Questions

**Problem: The program runs out of memory (OOM)**
**Solution:**

1. Reduce `-image_size` (e.g., to 512 or 256).
2. If using GPU, use `-backend cudnn`.
3. For video: Reduce the input video resolution before processing.

**Problem: Video processing is very slow**
**Solution:**
Video DeepDreaming is computationally expensive. It runs the full DeepDream process *per frame*, plus Optical Flow calculations.

* **Use `-num_iterations 1`** (recommended for video; temporal consistency means fewer iterations are needed).
* Reduce `-octave_iter` (e.g., to 10 or 20).
* Use a smaller `-image_size`.

## Memory Usage

By default, `neural-dream` uses the `nn` backend.

* **Use cuDNN**: `-backend cudnn` (GPU only, reduces memory).
* **Reduce Size**: `-image_size 256` (Halves memory usage).

With default settings, standard execution uses ~1.3 GB GPU memory.

## Multi-GPU Scaling

You can use multiple devices with `-gpu` and `-multidevice_strategy`.
Example: `-gpu 0,1,2,3 -multidevice_strategy 3,6,12` splits layers across 4 GPUs. See [ProGamerGov/neural-dream](https://github.com/ProGamerGov/neural-dream#multi-gpu-scaling) for details.
