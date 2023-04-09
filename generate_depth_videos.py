import cv2
import numpy as np
import os
import shutil
import time
import subprocess
import imageio
from tqdm import tqdm
from run import initialize_model, process_images
from pipeline import pipeline
from utils import depth_to_rgb24, batch_process, depth_to_rgb24_pixels, write_depth, scale_depth_values
import logging
logging.basicConfig(level=logging.ERROR)

saved_output = False

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
    video_filename = os.path.basename(video_path)
    output_video_filename = os.path.join(output_folder, video_filename.replace(os.path.splitext(video_filename)[1], '_output.mp4'))
    output_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-maxrate', '8M', '-bufsize', '512M']
    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{frame_size[0]}x{frame_size[1]}", '-r', str(fps), '-i', '-', '-i', video_path, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-muxdelay', '0', *output_args, output_video_filename]
    video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Output video size: {frame_size[1]} x {frame_size[0]}")
    print("ffmpeg command: ", ' '.join(ffmpeg_cmd))
    return video_writer, output_video_filename

def write_output_video(predictions, frames, video_writer):
    """Write combined frames to output video."""
    for i, prediction in enumerate(predictions):
        combined_frame = pipeline(prediction, frames[i])
        if combined_frame is not None:
            video_writer.stdin.write(combined_frame.tobytes())

def resize_frame(frame, min_width=384):
    height, width, _ = frame.shape
    if width >= min_width:
        return frame.copy()

    new_width = min_width
    aspect_ratio = float(height) / float(width)
    new_height = int(new_width * aspect_ratio)

    return cv2.resize(frame.copy(), (new_width, new_height), interpolation=cv2.INTER_AREA)


def resize_to_original_size(predictions, original_frame_size):
    resized_predictions = []

    for pred in predictions:
        resized_pred = cv2.resize(pred, original_frame_size[::-1], interpolation=cv2.INTER_AREA)
        resized_predictions.append(resized_pred)

    return resized_predictions

def process_video(video_path, output_folder, model_path, model_type, optimize=False, height=None, square=False, batch_size=8):
    # Initialize the model
    device, model, transform, net_w, net_h = initialize_model(model_path, model_type, optimize, height, square)

    # Initialize video reader and writer
    video_reader, fps, frame_size, frame_count = initialize_video(video_path)
    frame_size_original = frame_size

    # Double the width of the frame size to accommodate the pipeline function
    frame_size = (frame_size[0] * 2, frame_size[1])

    video_writer, output_video_filename = initialize_output(output_folder, frame_size, fps, video_path)

    # Initialize the tqdm progress bar with the total number of frames
    progress_bar = tqdm(total=frame_count, desc="Processing video", unit=" frames", position=0, leave=True)

    # Initialize time variables
    time_frame_processing = 0
    time_depth_prediction = 0
    time_output_video = 0

    # Iterate over video frames and process them in batches
    for i, frames in enumerate(batch_process(video_reader, batch_size)):
        # Resize the frames while maintaining the aspect ratio
        resized_frames = [resize_frame(frame) for frame in frames]

        # Process the resized frames and get depth predictions
        start_time = time.time()
        resized_frames = np.array(resized_frames, dtype=np.float64) / 255.0
        time_frame_processing += time.time() - start_time

        start_time = time.time()
        predictions = process_images(device, model, model_type, transform, net_w, net_h, resized_frames, optimize)
        time_depth_prediction += time.time() - start_time

        # Resize predictions and frames back to the original size
        predictions = resize_to_original_size(predictions, frame_size)

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


def is_video_file(filename):
    supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.wedbm', '.m4v', '.3gp']
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in supported_extensions

if __name__ == "__main__":
    output_folder = "output/"

    model_type = "dpt_beit_base_384"
    # model_type = "dpt_swin2_base_384"
    # model_type = "dpt_beit_large_512"

    model_path = f"weights/{model_type}.pt"
    optimize = True
    height = None
    square = False
    batch_size = 16

    input_folder = "unformatted_videos/"

    for filename in os.listdir(input_folder):
        if is_video_file(filename):
            video_path = os.path.join(input_folder, filename)
            print(f"Processing video file: {video_path}")
            process_video(video_path, output_folder, model_path, model_type, optimize, height, square, batch_size)
