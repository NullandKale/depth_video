import cv2
import numpy as np
import os
import math
import time
import torch
from typing import List
import imageio
from tqdm import tqdm
from run import initialize_model, process_images
from utils import depth_to_rgb24, batch_process, depth_to_rgb24_pixels, write_depth
import logging
import subprocess
logging.basicConfig(level=logging.ERROR)
import torch.nn.functional as F


class PipelineSettings:
    def __init__(self):
        self.name = "default"
        self.taa_strength = 0.35
        self.bins = 1024
        self.central_ratio = 0.8
        self.tss = 0.7
        self.max_background = 0.0
        self.peak_threshold = 0.95
        self.focus_percentage = 0.05
        self.max_tries = 1000
        self.decrement_ratio = 0.02
        self.b_color = 0.00
        self.bf_color = 0.5
        self.f_color = 0.75
        self.p_color = 0.95


def generate_settings_range(setting_name: str, min_value: float, max_value: float, steps: int) -> List[PipelineSettings]:
    if not hasattr(PipelineSettings(), setting_name):
        raise ValueError(f"Invalid setting name: {setting_name}")

    step_size = (max_value - min_value) / (steps - 1)
    settings_objects = []
    unique_values = set()  # Keep track of unique values

    for step in range(steps):
        value = min_value + step_size * step

        # Special handling for the "bins" setting
        if setting_name == "bins":
            # Snap to the closest power of 2
            value = int(2 ** round(np.log2(value)))

        # Skip if the value is already added
        if value in unique_values:
            continue

        unique_values.add(value)
        new_settings = PipelineSettings()
        new_settings.name = f"{setting_name}_{value}"
        setattr(new_settings, setting_name, value)
        settings_objects.append(new_settings)

    return settings_objects

def get_settings_count() -> int:
    setting_names = [attr for attr in vars(PipelineSettings()) if attr != "name"]

    return len(setting_names)

def get_setting_name_by_index(index: int) -> str:
    setting_names = [attr for attr in vars(PipelineSettings()) if attr != "name"]

    # Check if the index is within the valid range
    if index < 0 or index >= len(setting_names):
        raise IndexError("Index out of range")

    return setting_names[index]

def get_min() -> PipelineSettings:
    min_settings = PipelineSettings()
    min_settings.taa_strength = 0
    min_settings.bins = 512
    min_settings.central_ratio = 0.001
    min_settings.tss = 0
    min_settings.max_background = 0
    min_settings.peak_threshold = 0.5
    min_settings.focus_percentage = 0
    min_settings.max_tries = 1000
    min_settings.decrement_ratio = 0
    min_settings.b_color = 0
    min_settings.bf_color = 0
    min_settings.f_color = 0
    min_settings.p_color = 0
    return min_settings

def get_max() -> PipelineSettings:
    max_settings = PipelineSettings()
    max_settings.taa_strength = 1
    max_settings.bins = 8192
    max_settings.central_ratio = 0.999
    max_settings.tss = 1
    max_settings.max_background = 0.25
    max_settings.peak_threshold = 1.0
    max_settings.focus_percentage = 1
    max_settings.max_tries = 1000
    max_settings.decrement_ratio = 0
    max_settings.b_color = 0
    max_settings.bf_color = 0
    max_settings.f_color = 0
    max_settings.p_color = 0
    return max_settings

def get_step() -> PipelineSettings:
    step_settings = PipelineSettings()
    step_settings.taa_strength = 0
    step_settings.bins = 0
    step_settings.central_ratio = 5
    step_settings.tss = 5
    step_settings.max_background = 5
    step_settings.peak_threshold = 5
    step_settings.focus_percentage = 0
    step_settings.max_tries = 0
    step_settings.decrement_ratio = 0
    step_settings.b_color = 0
    step_settings.bf_color = 0
    step_settings.f_color = 0
    step_settings.p_color = 0
    return step_settings

"""
Pipeline Function Overview:

The `pipeline` and 'special_pipeline' functions are designed to perform post-processing on AI-generated depth video frames and then combine them with corresponding RGB video frames.

- Input: 
  1. Depth video frame (prediction): The depth information generated by an AI model.
  2. RGB video frame (color_frame): The corresponding color information from the original video.

- Processing:
  It first applies specific post-processing techniques to the depth frames, such as smoothing, normalization, or other transformations to enhance or modify the depth information.

- Combination:
  After post-processing the depth data, the function combines it with the RGB frames to create a single frame. This combined frame contains both color and depth information.

- Usage:
  This pipeline function is used within video processing workflows to transform a standard video into a format that includes depth information, suitable for 3D display.

The `pipeline` function is central to the creation of RGB+D videos, allowing for the visualization of depth alongside color in various applications, including 3D displays and augmented reality.
"""

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

def center_depth_average(depth_frame, target_center=32767.5):
    depth_avg = np.nanmean(depth_frame)

    # Calculate the shift needed to align the average with the target center value
    shift_value = target_center - depth_avg

    # Apply the shift to the entire depth frame
    centered_depth_frame = depth_frame + shift_value

    # Optionally, clip the values to ensure they stay within the valid range
    centered_depth_frame = np.clip(centered_depth_frame, 0, 65535)

    return centered_depth_frame

def expand_variance(depth_frame, target_std_dev=65535 * 0.2):
    # Calculate the current mean and standard deviation of the depth frame
    depth_mean = np.nanmean(depth_frame)
    depth_std_dev = np.nanstd(depth_frame)

    # Calculate the scaling factor to achieve the target standard deviation
    scaling_factor = target_std_dev / depth_std_dev

    # Scale the depth values and re-center them around the mean
    expanded_variance_depth_frame = (depth_frame - depth_mean) * scaling_factor + depth_mean

    # Optionally, clip the values to ensure they stay within the valid range
    expanded_variance_depth_frame = np.clip(expanded_variance_depth_frame, 0, 65535)

    return expanded_variance_depth_frame

def apply_taa_np(depth_frame, taa_strength=0.35):
    if apply_taa_np.previous_frame is None or apply_taa_np.previous_frame.shape != depth_frame.shape:
        apply_taa_np.previous_frame = depth_frame.copy()
    else:
        depth_frame = (taa_strength * apply_taa_np.previous_frame + (1 - taa_strength) * depth_frame)
        apply_taa_np.previous_frame = depth_frame.copy()
    return depth_frame

apply_taa_np.previous_frame = None

def update_thresholds(depth_frame, depth_max, bins=64, central_ratio=0.50, tss=0.8, max_background=0.05, peak_threshold=0.95, focus_percentage=0.05, max_tries=1000, decrement_ratio=0.02):
    threshold_value = depth_max * peak_threshold
    max_background_value = depth_max * max_background

    # Define the central region for sampling
    central_region = depth_frame[
        int(depth_frame.shape[0] * (1 - central_ratio) / 2):int(depth_frame.shape[0] * (1 + central_ratio) / 2),
        int(depth_frame.shape[1] * (1 - central_ratio) / 2):int(depth_frame.shape[1] * (1 + central_ratio) / 2)
    ]

    # Create a histogram and get bin edges using the central region
    histogram, bin_edges = np.histogram(central_region.flatten(), bins=bins, range=(0, depth_max))

    # Identify the new thresholds for background, focus, and foreground
    new_background_threshold = bin_edges[np.argmax(histogram[:bins//4])]
    if new_background_threshold > max_background_value:
        new_background_threshold = max_background_value
    
    focus_bin_start = np.argmax(histogram[bins//4:]) + bins//4
    new_focus_threshold = bin_edges[focus_bin_start]

    # Number of attempts to find the right focus threshold
    tries = 0

    # Ensure at least 5% of pixels are in the focal range
    decrement_step = (focus_bin_start - bins//4) * decrement_ratio
    while np.sum(histogram[focus_bin_start:bins//2]) / np.sum(histogram) < focus_percentage and focus_bin_start > bins//4 and tries < max_tries:
        focus_bin_start -= int(decrement_step)
        new_focus_threshold = bin_edges[focus_bin_start]
        tries += 1

    new_foreground_threshold = bin_edges[np.argmax(histogram[bins//2:]) + bins//2] if max(histogram[bins//2:]) > threshold_value else depth_max

    if(new_foreground_threshold > threshold_value):
        update_thresholds.has_foreground = True
    else:
        update_thresholds.has_foreground = False

    # Apply temporal stabilization to the thresholds
    update_thresholds.background_threshold = tss * update_thresholds.background_threshold + (1 - tss) * new_background_threshold
    update_thresholds.focus_threshold = tss * update_thresholds.focus_threshold + (1 - tss) * new_focus_threshold
    update_thresholds.foreground_threshold = tss * update_thresholds.foreground_threshold + (1 - tss) * new_foreground_threshold

    # Clamp the thresholds to be in the valid range [0, depth_max]
    update_thresholds.background_threshold = np.clip(update_thresholds.background_threshold, 0, depth_max)
    update_thresholds.focus_threshold = np.clip(update_thresholds.focus_threshold, 0, depth_max)
    update_thresholds.foreground_threshold = np.clip(update_thresholds.foreground_threshold, 0, depth_max)

    return update_thresholds.background_threshold, update_thresholds.focus_threshold, update_thresholds.foreground_threshold, update_thresholds.has_foreground

# Initialize the thresholds for temporal stabilization
update_thresholds.background_threshold = 0
update_thresholds.focus_threshold = 0
update_thresholds.foreground_threshold = 65535
update_thresholds.has_foreground = False

def create_segmented_depth_image(depth_frame, background_threshold, focus_threshold, foreground_threshold, has_foreground, b_color=0.00, bf_color=0.5, f_color=0.75, p_color=0.95):
    # Cast to float32 for consistency
    depth_frame = depth_frame.astype(np.float32)
    background_threshold = float(background_threshold)
    between_threshold = (background_threshold + focus_threshold) / 2
    focus_threshold = float(focus_threshold)
    foreground_threshold = float(foreground_threshold)

    # Define colors for the segments
    background_color = np.array([255 * b_color, 0 * b_color, 255 * b_color], dtype=np.float32)
    between_color = np.array([255 * bf_color, 255 * bf_color, 255 * bf_color], dtype=np.float32)
    focus_color = np.array([255 * f_color, 255 * f_color, 255 * f_color], dtype=np.float32)
    foreground_color = np.array([255 * p_color, 255 * p_color, 255 * p_color], dtype=np.float32)

    # Create an output image to store the segmented depth
    segmented_depth_image = np.zeros((*depth_frame.shape, 3), dtype=np.float32)

    # Create a threshold between background and focus
    between_threshold = (background_threshold + focus_threshold) / 2

    # Calculate gradient between background and the new between color
    background_between_mask = (depth_frame > background_threshold) & (depth_frame <= between_threshold)
    alpha_bb = (depth_frame[background_between_mask] - background_threshold) / (between_threshold - background_threshold)
    segmented_depth_image[background_between_mask] = (1 - alpha_bb)[:, None] * background_color + alpha_bb[:, None] * between_color

    # Calculate gradient between the new between color and focus
    between_focus_mask = (depth_frame > between_threshold) & (depth_frame <= focus_threshold)
    alpha_bf = (depth_frame[between_focus_mask] - between_threshold) / (focus_threshold - between_threshold)
    segmented_depth_image[between_focus_mask] = (1 - alpha_bf)[:, None] * between_color + alpha_bf[:, None] * focus_color

    # Calculate gradient between focus and foreground or depth_max
    if(has_foreground):
        focus_foreground_mask = depth_frame > focus_threshold
        alpha_ff = (depth_frame[focus_foreground_mask] - focus_threshold) / (foreground_threshold - focus_threshold)
        segmented_depth_image[focus_foreground_mask] = (1 - alpha_ff)[:, None] * focus_color + alpha_ff[:, None] * foreground_color

    # Clip the values to be in the valid range [0, 255]
    segmented_depth_image = np.clip(segmented_depth_image, 0, 255)

    # Cast the final result to uint8
    segmented_depth_image = segmented_depth_image.astype(np.uint8)

    return segmented_depth_image

def special_pipeline(prediction, color_frame, settings: PipelineSettings = None):
    if settings is None:
        settings = PipelineSettings()
    
    # Calculate depth range from prediction
    depth_max = 65535
    depth_avg = np.nanmean(prediction)

    if depth_avg == 0:
        prediction = apply_taa_np.previous_frame

    # Apply TAA
    depth_frame_np = apply_taa_np(prediction, settings.taa_strength)

    # Update thresholds
    background_threshold, focus_threshold, foreground_threshold, has_foreground = update_thresholds(
        depth_frame_np,
        depth_max,
        bins=settings.bins,
        central_ratio=settings.central_ratio,
        tss=settings.tss,
        max_background=settings.max_background,
        peak_threshold=settings.peak_threshold,
        focus_percentage=settings.focus_percentage,
        max_tries=settings.max_tries,
        decrement_ratio=settings.decrement_ratio
    )

    # Create segmented and colored depth image
    segmented_depth_image = create_segmented_depth_image(
        depth_frame_np,
        background_threshold,
        focus_threshold,
        foreground_threshold,
        has_foreground,
        b_color=settings.b_color,
        bf_color=settings.bf_color,
        f_color=settings.f_color,
        p_color=settings.p_color
    )

    # Convert to Torch tensor and resize
    segmented_depth_frame = to_torch(segmented_depth_image)
    segmented_depth_frame = F.interpolate(segmented_depth_frame, size=(color_frame.shape[0], color_frame.shape[1]), mode='nearest')

    # Combine color and depth frames
    color_frame_torch = to_torch(color_frame)
    combined_frame = torch.cat((color_frame_torch, segmented_depth_frame), dim=-1)
    return from_torch(combined_frame)

def pipeline(prediction, color_frame):
    global depth_min_buffer, depth_max_buffer

    # Calculate depth range from prediction
    depth_max = 65535
    depth_avg = np.nanmean(prediction)

    if depth_avg == 0:
        prediction = apply_taa_np.previous_frame
    
    # Apply TAA
    depth_frame_np = apply_taa_np(prediction, 0.15)

    # make the variance less wide so that the focal range is stable
    expanded_variance_depth_frame = expand_variance(depth_frame_np, depth_max * 0.1)

    # Center the average of the depth frame on the middle of the range so that we can just focus in the middle
    centered_depth_frame = center_depth_average(expanded_variance_depth_frame)

    # Scale depth values to range [0, 2^8-1] if range_diff is valid
    depth_scaled = (centered_depth_frame) / depth_max * ((2 ** 8) - 1)

    # Convert to uint8 and reshape to 2D image
    depth_image = np.uint8(depth_scaled).reshape(prediction.shape)

    # Convert to grayscale
    depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

    # Convert to Torch tensor
    depth_frame = to_torch(depth_gray)

    # Resize depth_frame to match color frame size
    depth_frame = F.interpolate(depth_frame, size=(color_frame.shape[0], color_frame.shape[1]), mode='nearest')

    # Combine color and depth frames
    color_frame_torch = to_torch(color_frame)
    combined_frame = torch.cat((color_frame_torch, depth_frame), dim=-1)
    return from_torch(combined_frame)