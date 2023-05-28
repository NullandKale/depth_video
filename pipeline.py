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

def apply_taa(depth_frame, taa_strength=0.5):
    if apply_taa.previous_frame is None or apply_taa.previous_frame.shape[-2:] != depth_frame.shape[-2:]:
        apply_taa.previous_frame = depth_frame.clone()
    else:
        depth_frame = (taa_strength * apply_taa.previous_frame + (1 - taa_strength) * depth_frame)
        apply_taa.previous_frame = depth_frame.clone()
    return apply_taa.previous_frame

apply_taa.previous_frame = None

def pipeline_depth_only(prediction, color_frame, enableTAA = False):
    # Convert prediction to a depth_frame using depth_to_rgb24
    depth_frame = depth_to_rgb24(prediction)
    depth_frame = to_torch(depth_frame)

    # Resize depth_frame to match color frame size
    depth_frame = F.interpolate(depth_frame, size=(color_frame.shape[0], color_frame.shape[1]), mode='nearest')

    # Apply TAA
    if enableTAA:
        depth_frame = apply_taa(depth_frame)

    return from_torch(depth_frame)


def pipeline(prediction, color_frame):
    # Convert prediction to a depth_frame using depth_to_rgb24
    depth_frame = depth_to_rgb24(prediction)
    depth_frame = to_torch(depth_frame)

    # Resize depth_frame to match color frame size
    depth_frame = F.interpolate(depth_frame, size=(color_frame.shape[0], color_frame.shape[1]), mode='nearest')

    # Apply TAA
    depth_frame = apply_taa(depth_frame)

    # Combine color and depth frames
    color_frame_torch = to_torch(color_frame)
    combined_frame = torch.cat((color_frame_torch, depth_frame), dim=-1)
    return from_torch(combined_frame)
