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

def segment_and_color_depth(depth_frame, depth_max, bins=32, central_ratio=0.50, tss=0.8, peak_threshold=0.9, b_color=0.05, bf_color=0.75, f_color=0.75, p_color=1.0, focus_percentage=0.05, max_tries=1000, decrement_ratio=0.02):
    threshold_value = depth_max * peak_threshold

    # Define the central region for sampling
    central_region = depth_frame[
        int(depth_frame.shape[0] * (1 - central_ratio) / 2):int(depth_frame.shape[0] * (1 + central_ratio) / 2),
        int(depth_frame.shape[1] * (1 - central_ratio) / 2):int(depth_frame.shape[1] * (1 + central_ratio) / 2)
    ]

    # Create a histogram and get bin edges using the central region
    histogram, bin_edges = np.histogram(central_region.flatten(), bins=bins, range=(0, depth_max))

    # Identify the new thresholds for background, focus, and foreground
    new_background_threshold = bin_edges[np.argmax(histogram[:bins//4])]
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

    new_foreground_threshold = bin_edges[np.argmax(histogram[bins//2:]) + bins//2] if max(histogram[bins//2:]) > threshold_value else None

    # Apply temporal stabilization to the thresholds
    segment_and_color_depth.background_threshold = tss * segment_and_color_depth.background_threshold + (1 - tss) * new_background_threshold
    segment_and_color_depth.focus_threshold = tss * segment_and_color_depth.focus_threshold + (1 - tss) * new_focus_threshold
    if new_foreground_threshold:
        segment_and_color_depth.foreground_threshold = tss * segment_and_color_depth.foreground_threshold + (1 - tss) * new_foreground_threshold

    # Define colors for the segments
    background_color = np.array([255 * b_color, 255 * b_color, 255 * b_color])
    between_color = np.array([255 * bf_color, 255 * bf_color, 255 * bf_color])
    focus_color = np.array([255 * f_color, 0, 255 * f_color])
    foreground_color = np.array([255 * p_color, 255 * p_color, 255 * p_color])

    # Create an output image to store the segmented depth
    segmented_depth_image = np.zeros((*depth_frame.shape, 3), dtype=np.uint8)

    # Create a threshold between background and focus
    between_threshold = (segment_and_color_depth.background_threshold + segment_and_color_depth.focus_threshold) / 2

    # Calculate gradient between background and the new between color
    background_between_mask = (depth_frame > segment_and_color_depth.background_threshold) & (depth_frame <= between_threshold)
    alpha_bb = (depth_frame[background_between_mask] - segment_and_color_depth.background_threshold) / (between_threshold - segment_and_color_depth.background_threshold)
    segmented_depth_image[background_between_mask] = (1 - alpha_bb)[:, None] * background_color + alpha_bb[:, None] * between_color

    # Calculate gradient between the new between color and focus
    between_focus_mask = (depth_frame > between_threshold) & (depth_frame <= segment_and_color_depth.focus_threshold)
    alpha_bf = (depth_frame[between_focus_mask] - between_threshold) / (segment_and_color_depth.focus_threshold - between_threshold)
    segmented_depth_image[between_focus_mask] = (1 - alpha_bf)[:, None] * between_color + alpha_bf[:, None] * focus_color

    # Calculate gradient between focus and foreground or depth_max
    focus_foreground_mask = depth_frame > segment_and_color_depth.focus_threshold
    if segment_and_color_depth.foreground_threshold:
        alpha_ff = (depth_frame[focus_foreground_mask] - segment_and_color_depth.focus_threshold) / (segment_and_color_depth.foreground_threshold - segment_and_color_depth.focus_threshold)
        segmented_depth_image[focus_foreground_mask] = (1 - alpha_ff)[:, None] * focus_color + alpha_ff[:, None] * foreground_color
    else:
        alpha_ff = (depth_frame[focus_foreground_mask] - segment_and_color_depth.focus_threshold) / (depth_max - segment_and_color_depth.focus_threshold)
        segmented_depth_image[focus_foreground_mask] = (1 - alpha_ff)[:, None] * focus_color + alpha_ff[:, None] * foreground_color

    return segmented_depth_image

# Initialize the thresholds for temporal stabilization
segment_and_color_depth.background_threshold = 0
segment_and_color_depth.focus_threshold = 0
segment_and_color_depth.foreground_threshold = 65535

def special_pipeline(prediction, color_frame):
    global depth_min_buffer, depth_max_buffer

    # Calculate depth range from prediction
    depth_max = 65535
    depth_avg = np.nanmean(prediction)

    if depth_avg == 0:
        prediction = apply_taa_np.previous_frame
    
    # Apply TAA
    depth_frame_np = apply_taa_np(prediction, 0.15)
 
    # Segment and color the depth frame
    segmented_depth_image = segment_and_color_depth(depth_frame_np, depth_max)

    # Convert to Torch tensor and resize
    segmented_depth_frame = to_torch(segmented_depth_image)
    segmented_depth_frame = F.interpolate(segmented_depth_frame, size=(color_frame.shape[0], color_frame.shape[1]), mode='nearest')

    # Combine color and depth frames
    color_frame_torch = to_torch(color_frame)
    combined_frame = torch.cat((color_frame_torch, segmented_depth_frame), dim=-1)
    return from_torch(combined_frame)

def pipeline(prediction, color_frame):
    global depth_min_buffer, depth_max_buffer
    centerAverage = True

    # Calculate depth range from prediction
    depth_max = 65535
    depth_avg = np.nanmean(prediction)

    if depth_avg == 0:
        prediction = apply_taa_np.previous_frame
    
    # Apply TAA
    depth_frame_np = apply_taa_np(prediction, 0.15)

    if centerAverage:
        # make the variance less wide so that the focal range is stable
        expanded_variance_depth_frame = expand_variance(depth_frame_np, depth_max * 0.1)

        # Center the average of the depth frame on the middle of the range so that we can just focus in the middle
        centered_depth_frame = center_depth_average(expanded_variance_depth_frame)

        # Scale depth values to range [0, 2^8-1] if range_diff is valid
        depth_scaled = (centered_depth_frame) / depth_max * ((2 ** 8) - 1)
    else:
        # Scale depth values to range [0, 2^8-1] if range_diff is valid
        depth_scaled = (depth_frame_np) / depth_max * ((2 ** 8) - 1)
        

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