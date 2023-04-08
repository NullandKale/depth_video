import os
import cv2
import time
import shutil
import itertools
import subprocess
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from collections import deque
from scipy.ndimage import gaussian_filter

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

def split_video_into_frames(input_video):
    output_folder = "./out/color_frames/"

    # Clear the output folder
    clear_folder(output_folder)

    # Call FFmpeg on the command line to split the video into PNGs
    cmd = f"ffmpeg -i {input_video} {output_folder}/frame_%07d.png"
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running FFmpeg: {e}")
    return output_folder

def generate_depth_from_frames(color_frames_folder):
    dest_folder = "./input/"
    output_folder = "./out/depth_frames/"
    depth_generation_command = "python run.py --model_type dpt_beit_large_512 --input_path input --output_path output"
    depth_generation_output_folder = "./output/"

    # Clear dest_folder
    clear_folder(dest_folder)

    # Move color frames into dest_folder
    for filename in os.listdir(color_frames_folder):
        shutil.copy(os.path.join(color_frames_folder, filename), dest_folder)

    # Run depthGenerationCommand on the command line and wait for it to finish
    subprocess.run(depth_generation_command, shell=True, check=True)

    # Clear output_folder
    clear_folder(output_folder)

    # Move files from depth_generation_output_folder to output_folder
    for filename in os.listdir(depth_generation_output_folder):
        if filename.endswith(".png"):
            src_path = os.path.join(depth_generation_output_folder, filename)
            dst_path = os.path.join(output_folder, filename)
            shutil.move(src_path, dst_path)

    return output_folder

def apply_bilateral_filter(frame, d=5, sigma_color=75, sigma_space=75):
    frame_32f = frame.astype(np.float32)
    filtered_frame = cv2.bilateralFilter(frame_32f, d, sigma_color, sigma_space)
    return filtered_frame


def read_frame(depth_frames_folder, file_name):
    frame = cv2.imread(os.path.join(depth_frames_folder, file_name), cv2.IMREAD_UNCHANGED)
    return frame.astype(np.float32) / 255


def apply_gaussian(frame, sigma):
    return gaussian_filter(frame, sigma=sigma, order=0, mode='reflect')


def calculate_difference(frame1, frame2):
    frame1_float = frame1.astype(np.float32)
    frame2_float = frame2.astype(np.float32)
    return np.abs(frame1_float - frame2_float)


def update_taa_buffer(curr_frame, taa_buffer, frame_threshold, dynamic_threshold, camera_cut_mult=0.85):
    if frame_threshold > dynamic_threshold:
        return (taa_buffer * (1.0 - camera_cut_mult)) + curr_frame * camera_cut_mult
    else:
        return (curr_frame + taa_buffer) / 2.0


def alpha_blending(curr_frame_smooth, taa_buffer, alpha):
    return (curr_frame_smooth * alpha) + (taa_buffer * (1 - alpha))

def depth_normalization(frame, min_val=0, max_val=1):
    frame = np.nan_to_num(frame)
    min_frame, max_frame = np.min(frame), np.max(frame)
    if max_frame == min_frame:
        return np.full(frame.shape, min_val, dtype=np.float32)
    normalized_frame = (frame - min_frame) * (max_val - min_val) / (max_frame - min_frame) + min_val
    return normalized_frame.astype(np.float32)

def threshold_otsu(image):
    epsilon = 1e-10

    # Compute the histogram of the image
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))

    # Normalize the histogram
    histogram = histogram.astype(float) / (np.sum(histogram) + epsilon)

    # Compute cumulative sums
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256))

    # Compute the global mean
    global_mean = cumulative_mean[-1]

    # Compute between-class variance
    between_class_variance = (global_mean * cumulative_sum - cumulative_mean) ** 2 / (cumulative_sum * (1 - cumulative_sum) + epsilon)

    # Find the maximum between-class variance
    max_variance_idx = np.argmax(between_class_variance)

    # Return the threshold corresponding to the maximum between-class variance
    return max_variance_idx

def edge_average_color_without_center(image, edge_percentage, center_percentage, threshold):
    edge_region_pixels = edge_region(image, edge_percentage)
    center_average_color = center_region(image, center_percentage)

    # Exclude values around the center_average_color using the threshold
    mask = np.abs(edge_region_pixels - center_average_color) > threshold
    filtered_edge_region_pixels = edge_region_pixels[mask]

    return np.mean(filtered_edge_region_pixels)

def edge_region(image, edge_percentage):
    height, width = image.shape
    edge_width = int(edge_percentage * width)
    edge_height = int(edge_percentage * height)

    top = image[:edge_height, :]
    bottom = image[-edge_height:, :]
    left = image[edge_height:-edge_height, :edge_width]
    right = image[edge_height:-edge_height, -edge_width:]

    top_flat = top.flatten()
    bottom_flat = bottom.flatten()
    left_flat = left.flatten()
    right_flat = right.flatten()

    combined = np.concatenate((top_flat, bottom_flat, left_flat, right_flat))

    # Reshape the combined array into a 2D array
    reshaped = combined.reshape(-1, 1)

    return reshaped

def center_region(image, center_percentage):
    center_width = int(min(image.shape[0], image.shape[1]) * center_percentage)
    h, w = image.shape[:2]
    top_left = (h // 2 - center_width // 2, w // 2 - center_width // 2)
    center_area = image[top_left[0]:top_left[0] + center_width, top_left[1]:top_left[1] + center_width]

    return np.mean(center_area)

def map_values_to_alpha(frame, center_average_color, edge_average_color):
    min_val = min(edge_average_color, center_average_color)
    max_val = max(edge_average_color, center_average_color)

    # Map the values of the frame to the range [0, 1]
    mapped_frame = (frame - min_val) / (max_val - min_val)

    # Make sure the values are within the range [0, 1]
    mapped_frame = np.clip(mapped_frame, 0, 1)

    # Calculate the center point between min_val and max_val
    center_point = 0.5 * (min_val + max_val)

    # Make values close to the center_average_color closer to 0.5
    mapped_frame = np.where(frame < center_point, 0.5 * mapped_frame, 1 - 0.5 * (1 - mapped_frame))

    return mapped_frame

def bidirectional_temporal_filter(current_frame, taa_buffer, alpha, filter_size):
    """
    Perform bidirectional temporal filtering using a fixed-size temporal window.
    
    :param current_frame: The current depth frame to be filtered.
    :param taa_buffer: The accumulated temporal anti-aliasing buffer.
    :param alpha: The blending factor.
    :param filter_size: The size of the temporal window.
    :return: The filtered frame.
    """
    filtered_frames = [current_frame]
    weights = [1]

    # Forward pass
    forward_frame = taa_buffer.copy()
    for _ in range(filter_size):
        forward_frame = alpha * current_frame + (1 - alpha) * forward_frame
        filtered_frames.append(forward_frame)
        weights.append((1 - alpha) ** (_ + 1))

    # Backward pass
    backward_frame = taa_buffer.copy()
    for _ in range(filter_size):
        backward_frame = alpha * current_frame + (1 - alpha) * backward_frame
        filtered_frames.append(backward_frame)
        weights.append((1 - alpha) ** (_ + 1))

    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # Weighted average
    filtered_frame = np.zeros_like(current_frame, dtype=np.float32)
    for frame, weight in zip(filtered_frames, weights):
        filtered_frame += weight * frame

    return filtered_frame

def unidirectional_temporal_filter(current_frame, taa_buffer, alpha, filter_size):
    """
    Perform unidirectional temporal filtering using a fixed-size temporal window.
    
    :param current_frame: The current depth frame to be filtered.
    :param taa_buffer: The accumulated temporal anti-aliasing buffer.
    :param alpha: The blending factor.
    :param filter_size: The size of the temporal window.
    :return: The filtered frame.
    """
    filtered_frames = [current_frame]
    weights = [1]

    # Forward pass
    forward_frame = taa_buffer.copy()
    for _ in range(filter_size):
        forward_frame = alpha * current_frame + (1 - alpha) * forward_frame
        filtered_frames.append(forward_frame)
        weights.append((1 - alpha) ** (_ + 1))

    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # Weighted average
    filtered_frame = np.zeros_like(current_frame, dtype=np.float32)
    for frame, weight in zip(filtered_frames, weights):
        filtered_frame += weight * frame

    return filtered_frame

def apply_TAA(depth_frames_folder, sigma=1.5, filter_size=5, bidirectional=True, num_levels=4):
    file_list = sorted(os.listdir(depth_frames_folder))
    num_files = len(file_list)

    # Initialize the TAA buffer with the first frame
    taa_buffer = read_frame(depth_frames_folder, file_list[0])

    # Get the target size from the first frame
    target_size = (taa_buffer.shape[1], taa_buffer.shape[0])

    # Apply TAA to each subsequent frame and save the result to disk
    for i in tqdm(range(1, num_files), desc="Applying TAA"):
        # Read the current frame
        curr_frame = read_frame(depth_frames_folder, file_list[i])

        # Resize the current frame to the target size
        curr_frame = cv2.resize(curr_frame, target_size)

        # Apply Gaussian smoothing to the current frame
        curr_frame_smooth = apply_gaussian(curr_frame, sigma)

        # Calculate the average color of the edge region
        # this is attempting to be a heuristic to what is not the region to focus on
        edge_average_color = edge_average_color_without_center(curr_frame_smooth, edge_percentage=0.1, center_percentage=0.5, threshold=10)

        # Calculate the average color of the center region
        # this is attempting to be a heuristic to what is the region to focus on 
        center_average_color = center_region(curr_frame_smooth, center_percentage=0.5)

        mapped_frame = map_values_to_alpha(curr_frame_smooth, center_average_color, edge_average_color)

        # Calculate the difference between the current frame and the TAA buffer
        frame_difference = calculate_difference(mapped_frame, taa_buffer)

        # Normalize the frame difference
        frame_difference_normalized = frame_difference / np.max(frame_difference)

        # Calculate alpha based on the mean of the normalized frame difference
        alpha = np.mean(frame_difference_normalized)

        # Update the TAA buffer with advanced temporal filtering
        if bidirectional:
            taa_buffer = bidirectional_temporal_filter(curr_frame_smooth, taa_buffer, alpha, filter_size)
        else:
            taa_buffer = unidirectional_temporal_filter(curr_frame_smooth, taa_buffer, alpha, filter_size)

        # Normalize pixel values within a specific range
        output_frame = depth_normalization(taa_buffer)

        # Convert the output frame back to the range 0-255 before saving
        output_frame = (output_frame * 255).astype(np.uint8)

        # Save the output frame to disk
        output_file = os.path.join(depth_frames_folder, file_list[i])
        cv2.imwrite(output_file, output_frame)

# Define a function to process each frame
def combine_frame(color_file, depth_file, color_frames_folder, depth_frames_folder, combined_frames_folder):
    color_frame = cv2.imread(os.path.join(color_frames_folder, color_file))
    depth_frame = cv2.imread(os.path.join(depth_frames_folder, depth_file))

    # Resize depth_frame to match color_frame dimensions
    depth_frame_resized = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    combined_frame = np.concatenate((color_frame, depth_frame_resized), axis=1)
    combined_file = os.path.join(combined_frames_folder, color_file)
    cv2.imwrite(combined_file, combined_frame)

def combine_frames_into_video(input_video, color_frames_folder, depth_frames_folder, output_video):
    # Combine the color and depth frames into a single video and save it as output_video

    # Create a folder to store the combined frames
    combined_frames_folder = './out/combined_frames/'
    clear_folder(combined_frames_folder)

    # Iterate through the files in both folders and combine them side by side
    color_files = sorted(os.listdir(color_frames_folder))
    depth_files = sorted(os.listdir(depth_frames_folder))

    # Use multiple processes to combine frames in parallel
    with multiprocessing.Pool() as pool:
        results = []
        for color_file, depth_file in zip(color_files, depth_files):
            result = pool.apply_async(combine_frame, (color_file, depth_file, color_frames_folder, depth_frames_folder, combined_frames_folder))
            results.append(result)
        
        # Wait for all processes to finish
        for result in tqdm(results, total=len(results), desc="Combining frames"):
            result.get()

        ffprobe_cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 "{input_video}"'
        frame_rate = subprocess.check_output(ffprobe_cmd, shell=True, text=True).strip()

        # Use ffmpeg to create a video from the combined frames
        input_pattern = os.path.join(combined_frames_folder, 'frame_%07d.png')
        command = f'ffmpeg -y -r {frame_rate} -i "{input_pattern}" -i "{input_video}" -c:v hevc_nvenc -pix_fmt yuv420p -preset fast -crf 28 -map 0:v -map 1:a? -c:a copy -shortest "{output_video}"'
    
    # Run the command and redirect the output to subprocess.PIPE
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the command to finish and get the output and errors
    output, errors = proc.communicate()

    # If there were any errors, print them
    if proc.returncode != 0:
        print(errors.decode())

def generate_depth_video(input_video, output_video):
    cuts_folder = "./cuts/"
    clear_folder(cuts_folder)
    # Split video into individual frames
    main_start_time = time.time()
    start_time = time.time()
    color_frames_folder = split_video_into_frames(input_video)
    elapsed_time = time.time() - start_time
    print(f"Splitting video into frames took {elapsed_time:.2f} seconds.")

    # Generate depth maps from the frames
    start_time = time.time()
    depth_frames_folder = generate_depth_from_frames(color_frames_folder)
    elapsed_time = time.time() - start_time
    print(f"Generating depth maps took {elapsed_time:.2f} seconds.")

    # Apply temporal smoothing to the depth maps
    start_time = time.time()
    apply_TAA(depth_frames_folder)

    elapsed_time = time.time() - start_time
    print(f"Applying temporal smoothing took {elapsed_time:.2f} seconds.")

    # Combine the color and depth frames into a single video
    start_time = time.time()
    combine_frames_into_video(input_video, color_frames_folder, depth_frames_folder, output_video)
    elapsed_time = time.time() - start_time
    total_elapsed_time = time.time() - main_start_time
    print(f"Combining frames into video took {elapsed_time:.2f} seconds.")
    print(f"Total time processing video took {total_elapsed_time:.2f} seconds.")

def convert_to_mp4(input_video, output_video, duration=None):
    duration_option = f'-t {duration}' if duration else ''
    command = f'ffmpeg -y {duration_option} -i "{input_video}" -c:v libx264 -preset fast -crf 28 -c:a aac "{output_video}"'
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = proc.communicate()

def process_all_videos(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_formats = [".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"]

    for video_file in os.listdir(unformatted_video_folder):
        file_extension = os.path.splitext(video_file)[-1].lower()
        if file_extension in supported_formats:
            input_video = os.path.join(unformatted_video_folder, video_file)
            converted_video = os.path.join(video_folder, os.path.splitext(video_file)[0] + ".mp4")
            print(f"Converting video: {input_video} to {converted_video}")
            convert_to_mp4(input_video, converted_video)
            output_video = os.path.join(output_folder, "depth_" + os.path.splitext(video_file)[0] + ".mp4")
            print(f"Processing video: {converted_video}")
            generate_depth_video(converted_video, output_video)
            #generate_depth_video_cuts(converted_video, output_video)


if __name__ == "__main__":
    unformatted_video_folder = "./unformatted_videos/"
    video_folder = "./videos/"
    output_folder = "./depth_videos/"
    process_all_videos(video_folder, output_folder)