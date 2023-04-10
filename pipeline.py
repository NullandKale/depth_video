import cv2
import numpy as np
import os
import math
import time
import torch
import imageio
from tqdm import tqdm
from run import initialize_model, process_images
from utils import depth_to_rgb24, batch_process, depth_to_rgb24_pixels, write_depth
import logging
import subprocess
logging.basicConfig(level=logging.ERROR)
import torch.nn.functional as F

def to_torch(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

def from_torch(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

def bias_depth_frame(depth_frame, bias_center=0.5, scale_factor=2.0):
    depth_frame = depth_frame / 255.0
    depth_frame = (torch.tanh((depth_frame - bias_center) * scale_factor) + 1.0) / 2.0
    depth_frame = (depth_frame * 255)
    return depth_frame

def apply_taa(depth_frame, taa_strength=0.5):
    if apply_taa.previous_frame is None:
        apply_taa.previous_frame = depth_frame.clone()
    taa_frame = (taa_strength * apply_taa.previous_frame + (1 - taa_strength) * depth_frame)
    apply_taa.previous_frame = depth_frame.clone()
    return taa_frame

apply_taa.previous_frame = None

def pipeline(prediction, color_frame, mode="fast"):
    # Convert prediction to a depth_frame using depth_to_rgb24
    depth_frame = depth_to_rgb24(prediction)
    depth_frame = to_torch(depth_frame)
    
    if mode == 'slow':
        # Bias depth_frame towards 0.5
        depth_frame = bias_depth_frame(depth_frame)

    # Apply TAA
    depth_frame = apply_taa(depth_frame)

    # Resize to match color frame
    resized_frame = F.interpolate(depth_frame, size=(color_frame.shape[0], color_frame.shape[1]), mode='nearest')

    # Combine color and depth frames
    color_frame_torch = to_torch(color_frame)
    combined_frame = torch.cat((color_frame_torch, resized_frame), dim=-1)
    return from_torch(combined_frame)