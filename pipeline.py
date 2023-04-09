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

def apply_temporal_median_filter(depth_frame, buffer_size=5):
    if apply_temporal_median_filter.buffer is None:
        apply_temporal_median_filter.buffer = [depth_frame.copy()] * buffer_size

    apply_temporal_median_filter.buffer.pop(0)
    apply_temporal_median_filter.buffer.append(depth_frame.copy())
    
    median_frame = np.median(apply_temporal_median_filter.buffer, axis=0).astype(np.uint8)
    return median_frame

apply_temporal_median_filter.buffer = None

def apply_bilateral_filter(depth_frame, diameter=5, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(depth_frame, diameter, sigma_color, sigma_space)

def bias_depth_frame(depth_frame, bias_center=0.5, scale_factor=2.0):
    depth_frame = depth_frame.astype(np.float32) / 255.0
    depth_frame = (np.tanh((depth_frame - bias_center) * scale_factor) + 1.0) / 2.0
    depth_frame = (depth_frame * 255).astype(np.uint8)
    return depth_frame

def apply_taa(depth_frame, taa_strength=0.5):
    if apply_taa.previous_frame is None:
        apply_taa.previous_frame = depth_frame.copy()
    taa_frame = (taa_strength * apply_taa.previous_frame + (1 - taa_strength) * depth_frame).astype(np.uint8)
    apply_taa.previous_frame = depth_frame.copy()
    return taa_frame

def apply_dilation(depth_frame, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(depth_frame, kernel, iterations=iterations)

apply_taa.previous_frame = None

def remove_background_gradient(depth_frame, threshold=128):
    _, depth_frame = cv2.threshold(depth_frame, threshold, 255, cv2.THRESH_BINARY)
    return depth_frame

def pipeline(prediction, color_frame, taa_strength=0.5, background_threshold=128):
    # Use depth_to_rgb24_pixels to convert prediction to a depth_frame
    depth_frame = depth_to_rgb24(prediction)

    # Bias depth_frame towards 0.5
    depth_frame = bias_depth_frame(depth_frame)

    # Apply Bilateral Filter
    depth_frame = apply_bilateral_filter(depth_frame)

    # Apply Dilation to expand bright edges
    depth_frame = apply_dilation(depth_frame, kernel_size=3, iterations=1)

    # Remove background gradient
    # depth_frame = remove_background_gradient(depth_frame, threshold=background_threshold)

    # Apply TAA
    depth_frame = apply_taa(depth_frame, taa_strength=taa_strength)

    # Resize to match color frame
    resized_frame = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Combine color and depth frames
    combined_frame = np.concatenate((color_frame, resized_frame), axis=1)
    return combined_frame
