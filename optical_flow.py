import torch
import numpy as np
import cv2
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def init_raft():
    device = get_device()
    print(f"Flow estimator using device: {device}")
    print("Loading RAFT model...")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=False).to(device)
    model.eval()
    transforms = weights.transforms()
    print("RAFT model loaded.")
    return model, transforms, device

def flow_to_image(flow_array):
    h, w = flow_array.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_array[..., 0], flow_array[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def warp_image(img, flow):
    h, w = flow.shape[:2]
    
    flow_map = np.column_stack((np.repeat(np.arange(h), w), np.tile(np.arange(w), h)))
    flow_map = flow_map.reshape((h, w, 2)).astype(np.float32)

    map_x = (flow_map[..., 1] - flow[..., 0]).astype(np.float32)
    map_y = (flow_map[..., 0] - flow[..., 1]).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

def estimate_flow(prev_frame_bgr, curr_frame_bgr, model, transforms, device):
    prev_rgb = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
    curr_rgb = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)

    prev_tensor = torch.from_numpy(prev_rgb).permute(2, 0, 1)
    curr_tensor = torch.from_numpy(curr_rgb).permute(2, 0, 1)

    img1_batch, img2_batch = transforms(
        prev_tensor.unsqueeze(0), 
        curr_tensor.unsqueeze(0)
    )
    
    img1_batch = img1_batch.to(device)
    img2_batch = img2_batch.to(device)

    with torch.no_grad():
        list_of_flows = model(img1_batch, img2_batch)
        predicted_flow = list_of_flows[-1][0]

    flow_np = predicted_flow.permute(1, 2, 0).cpu().numpy()
    flow_vis = flow_to_image(flow_np)

    return flow_np, flow_vis

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