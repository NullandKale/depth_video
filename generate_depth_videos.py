import cv2
import numpy as np
import os
import shutil
import time
import subprocess
import signal
import threading
import imageio
from tqdm import tqdm
from run import initialize_model, process_images
from pipeline import pipeline, special_pipeline, PipelineSettings
from output_adapter import initialize_output, write_output_video
from utils import depth_to_rgb24, batch_process, depth_to_rgb24_pixels, write_depth
import logging
logging.basicConfig(level=logging.ERROR)

saved_output = False
video_writer = None

def signal_handler(sig, frame):
    # Close the video writer
    video_writer.close()
    exit(0)

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

def process_video(video_path, output_folder, model_path, model_type, optimize, height, square, batch_size, settings : PipelineSettings = None):
    # Initialize the model
    model = initialize_model(model_path, model_type, optimize, height, square)

    # Initialize video reader and writer
    video_reader, fps, frame_size, frame_count = initialize_video(video_path)

    # Double the width of the frame size to accommodate the pipeline function
    frame_size = (frame_size[0] * 2, frame_size[1])

    if settings != None:
        override_name = settings.name
    else:
        override_name = None

    video_writer = initialize_output(output_folder, frame_size, fps, video_path, override_name, output_mode='file')

    # Initialize the tqdm progress bar with the total number of frames
    progress_bar = tqdm(total=frame_count, desc=f"Processing video with settings {override_name}", unit=" frames", position=0, leave=True)

    # Initialize time variables
    time_depth_prediction = 0
    time_output_video = 0
    time_pipeline = 0

    def write_frames(predictions, frames):
        nonlocal time_output_video, time_pipeline
        start_time = time.time()
        for prediction, frame in zip(predictions, frames):
            pipeline_start = time.time()
            combined_frame = special_pipeline(prediction, frame, settings)
            time_pipeline += time.time() - pipeline_start
            write_output_video(combined_frame, video_writer)
        time_output_video += time.time() - start_time
        progress_bar.update(len(frames))

    write_frames_thread = None

    for _, frames in enumerate(batch_process(video_reader, batch_size)):
        # Generate depth frames
        start_time = time.time()
        predictions = process_images(model, frames, optimize, min_width=frame_size[1])
        time_depth_prediction += time.time() - start_time

        # Wait for previous write_frames to complete if it's still running
        if write_frames_thread is not None:
            write_frames_thread.join()

        # Start write_frames in a new thread
        write_frames_thread = threading.Thread(target=write_frames, args=(predictions, frames))
        write_frames_thread.start()

    # Wait for the last write_frames thread to complete
    if write_frames_thread is not None:
        write_frames_thread.join()

    video_writer.close()

    frame_count = float(frame_count)
    print(f"\n\nFinished Generating: {video_writer.output_video_filename}")
    print(f"Time taken for depth prediction: {(time_depth_prediction * 1000) / frame_count:.2f} ms per frame with a batch_size of {batch_size}")
    print(f"Time taken for writing output video: {(time_output_video * 1000) / frame_count:.2f} ms per frame")
    print(f"Time taken for pipeline: {(time_pipeline * 1000) / frame_count:.2f} ms per frame")
    return os.path.abspath(video_writer.output_video_filename)