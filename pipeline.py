import cv2
import numpy as np
import os
import math
import time
import torch
import imageio
from tqdm import tqdm
from run import initialize_model, process_images
from utils import depth_to_rgb24, batch_process, depth_to_rgb24_pixels, write_depth, scale_depth_values
import logging
import subprocess
logging.basicConfig(level=logging.ERROR)

def apply_taa(depth_frame, taa_strength=0.5):
    if apply_taa.previous_frame is None:
        apply_taa.previous_frame = depth_frame.copy()
    taa_frame = (taa_strength * apply_taa.previous_frame + (1 - taa_strength) * depth_frame).astype(np.uint8)
    apply_taa.previous_frame = depth_frame.copy()
    return taa_frame

apply_taa.previous_frame = None

def pipeline(prediction, color_frame, taa_strength=0.5):
    # Use depth_to_rgb24_pixels to convert prediction to a depth_frame
    depth_frame = depth_to_rgb24(prediction)

    # Apply TAA to depth frame
    taa_frame = apply_taa(depth_frame, taa_strength=taa_strength)

    # Resize to match color frame
    resized_frame = cv2.resize(taa_frame, (color_frame.shape[1], color_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert color frame from RGB to BGR format for cv2
    color_frame = (color_frame * 255).astype(np.uint8)
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

    # Combine color and depth frames
    combined_frame = np.concatenate((color_frame, resized_frame), axis=1)
    return combined_frame
