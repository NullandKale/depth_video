import os
import subprocess
import random
import shutil

def is_video_file(filename):
    supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp']
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in supported_extensions

def get_video_duration(video_path):
    try:
        # Using ffprobe to get the duration of the video in seconds
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Split the output by lines and take the first line, which contains the duration value
        duration_str = result.stdout.decode().splitlines()[0]
        
        return float(duration_str)
    except Exception as e:
        print(f"An error occurred while getting the duration for {video_path}: {e}")
        return -1

def extract_random_minute(video_path, output_path):
    # Get the duration of the video
    duration = get_video_duration(video_path)

    # Select a random start time for a 60-second clip, making sure it's at least 60 seconds from the end
    start_time = random.uniform(0, duration - 60)

    # Using ffmpeg to extract 1 random minute from the video
    subprocess.run(["ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", "60", "-c:v", "copy", output_path])
    # print(["ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", "60", output_path])

def process_videos(search_folder1, search_folder2, search_output_folder, count):
    video_files = []
    
    # Function to search for video files in a given folder
    def search_videos_in_folder(search_folder):
        for root, _, files in os.walk(search_folder):
            for filename in files:
                if is_video_file(filename):
                    video_path = os.path.join(root, filename)
                    video_files.append(video_path)

    # Search recursively for video files in both input folders
    search_videos_in_folder(search_folder1)
    # search_videos_in_folder(search_folder2)

    # Shuffle the list of video files to ensure randomness
    random.shuffle(video_files)

    # Extract 1 random minute from each selected video, only considering videos longer than 10 minutes
    selected_count = 0
    for video_path in video_files:
        if get_video_duration(video_path) > 600: # Check if the video is longer than 10 minutes
            output_filename = os.path.join(search_output_folder, os.path.basename(video_path))
            extract_random_minute(video_path, output_filename)
            selected_count += 1
            if selected_count == count:
                break

if __name__ == "__main__":
    search_folder1 = "D:/Videos/Movies/"
    search_folder2 = "D:/Videos/TV/"
    search_output_folder = "test_videos/"
    process_videos(search_folder1, search_folder2, search_output_folder, 60)
