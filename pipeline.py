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

saved_output = False

def clear_output_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def get_metadata(video_path):
    """Initialize video reader and get metadata."""
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_capture.release()
    return fps, frame_size, frame_count

def initialize_video(video_path):
    """Initialize video reader and get metadata."""
    fps, frame_size, frame_count = get_metadata(video_path)
    video_reader = imageio.get_reader(video_path)
    print(f"Input video size: {frame_size[1]} x {frame_size[0]}")
    return video_reader, fps, frame_size, frame_count

def initialize_output(output_folder, frame_size, fps, video_path):
    """Initialize output folder and video writer."""
    os.makedirs(output_folder, exist_ok=True)
    output_video_filename = os.path.join(output_folder, 'output_video.mp4')
    output_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-maxrate', '4M', '-bufsize', '8M']
    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{frame_size[0]}x{frame_size[1]}", '-r', str(fps), '-i', '-', '-i', video_path, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-muxdelay', '0', *output_args, output_video_filename]
    video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Output video size: {frame_size[1]} x {frame_size[0]}")
    print("ffmpeg command: ", ' '.join(ffmpeg_cmd))
    return video_writer, output_video_filename

def pipeline(prediction, color_frame):
    # Use depth_to_rgb24_pixels to convert prediction to a depth_frame
    depth_frame = depth_to_rgb24(prediction)
    global saved_output
    if saved_output == False:
        saved_output = True
        write_depth("out/test.png", prediction)

    # Resize to match color frame
    depth_frame = cv2.resize(depth_frame, (color_frame.shape[1], color_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert color frame from RGB to BGR format for cv2
    color_frame = (color_frame * 255).astype(np.uint8)
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)

    # Combine color and depth frames
    combined_frame = np.concatenate((color_frame, depth_frame), axis=1)
    return combined_frame

def write_output_video(predictions, frames, video_writer):
    """Write combined frames to output video."""
    for i, prediction in enumerate(predictions):
        combined_frame = pipeline(prediction, frames[i])
        if combined_frame is not None:
            video_writer.stdin.write(combined_frame.tobytes())

def process_video(video_path, output_folder, model_path, model_type, optimize=False, height=None, square=False, batch_size=8):
    # Initialize the model
    device, model, transform, net_w, net_h = initialize_model(model_path, model_type, optimize, height, square)

    # Initialize video reader and writer
    video_reader, fps, frame_size, frame_count = initialize_video(video_path)

    # Double the width of the frame size to accommodate the pipeline function
    frame_size = (frame_size[0]*2, frame_size[1])

    video_writer, output_video_filename = initialize_output(output_folder, frame_size, fps, video_path)

    # Initialize the tqdm progress bar with the total number of frames
    progress_bar = tqdm(total=frame_count, desc="Processing video", unit=" frames", position=0, leave=True)

    # Initialize time variables
    time_frame_processing = 0
    time_depth_prediction = 0
    time_output_video = 0

    # Iterate over video frames and process them in batches
    for i, frames in enumerate(batch_process(video_reader, batch_size)):
        # Process the frames and get depth predictions
        start_time = time.time()
        frames = np.array(frames, dtype=np.float64) / 255.0
        time_frame_processing += time.time() - start_time

        start_time = time.time()
        predictions = process_images(device, model, model_type, transform, net_w, net_h, frames, optimize)
        time_depth_prediction += time.time() - start_time

        # Write combined frames to output video
        start_time = time.time()
        write_output_video(predictions, frames, video_writer)
        time_output_video += time.time() - start_time

        # Update the tqdm progress bar with the number of frames processed in this batch
        progress_bar.update(len(frames))

    # Close the video reader and writer
    video_reader.close()
    video_writer.stdin.close()
    video_writer.wait()

    print(f"Video processing complete. Combined video saved as {output_video_filename}")
    print(f"Time taken for frame processing: {time_frame_processing:.2f} seconds")
    print(f"Time taken for depth prediction: {time_depth_prediction:.2f} seconds with a batch_size of {batch_size}")
    print(f"Time taken for writing output video: {time_output_video:.2f} seconds")



if __name__ == "__main__":
    
    output_folder = "out/depth_videos/"
    
    model_type = "dpt_beit_base_384"
    #model_type = "dpt_beit_large_384"
    #model_type = "dpt_beit_large_512"

    model_path = f"weights/{model_type}.pt"
    optimize = True
    height = None
    square = False
    batch_size = 16

    # Clear the output folder before rendering
    clear_output_folder(output_folder)

    video_path = "unformatted_videos/output.mp4"
    process_video(video_path, output_folder, model_path, model_type, optimize, height, square, batch_size)
