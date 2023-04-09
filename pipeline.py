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

def pipeline(prediction, color_frame, taa_strength=0.5):
    # Use depth_to_rgb24_pixels to convert prediction to a depth_frame
    depth_frame = depth_to_rgb24(prediction)

    # Resize to match color frame
    depth_frame = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply TAA to depth frame
    if 'previous_frame' not in pipeline.__dict__:
        pipeline.previous_frame = depth_frame
    depth_frame = (taa_strength * pipeline.previous_frame + (1 - taa_strength) * depth_frame).astype(np.uint8)
    pipeline.previous_frame = depth_frame.copy()

    # Convert color frame from RGB to BGR format for cv2
    color_frame = (color_frame * 255).astype(np.uint8)
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

    # Combine color and depth frames
    combined_frame = np.concatenate((color_frame, depth_frame), axis=1)
    return combined_frame
