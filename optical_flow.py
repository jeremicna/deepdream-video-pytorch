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

def flow_to_image(flow_tensor):
    flow = flow_tensor.permute(1, 2, 0).cpu().numpy()
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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

    return flow_to_image(predicted_flow)