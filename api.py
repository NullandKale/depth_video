import os
import uuid
import queue
import shutil
import threading
from generate_depth_videos import process_video as depth_gen

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder="src")
CORS(app)  # Add this line to enable CORS

video_root = "D:/Videos/TV"

output_folder = "output/"
cache_folder = "G:/Cache/"

cache_file = "G:/CacheFile.text"

model_type = "dpt_beit_base_384"
# model_type = "dpt_beit_large_384"
# model_type = "dpt_next_vit_large_384"
# model_type = "dpt_swin2_base_384"
# model_type = "dpt_beit_large_512"

model_path = f"weights/{model_type}.pt"

optimize = True
height = None
square = False
batch_size = 32

# Create a queue to store video paths for processing
video_queue = queue.Queue()

# Create a dictionary to store processed video paths
processed_videos = {}
in_progress_videos = set()


def is_video_file(filename):
    supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.wedbm', '.m4v', '.3gp']
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in supported_extensions

def init():
    app.run(debug=True)

def search_test():
    test_data = {
        "Show 1": {
            "Season 1": ["Episode 1.mp4", "Episode 2.mp4", "Episode 3.mp4"],
            "Season 2": ["Episode 1.mp4", "Episode 2.mp4", "Episode 3.mp4"]
        },
        "Show 2": {
            "Season 1": ["Episode 1.mp4", "Episode 2.mp4", "Episode 3.mp4"],
            "Season 2": ["Episode 1.mp4", "Episode 2.mp4", "Episode 3.mp4"]
        } 
    }
    return test_data

def search(video_root):
    shows = {}
    for show in os.listdir(video_root):
        show_path = os.path.join(video_root, show)
        if os.path.isdir(show_path):
            seasons = {}
            for entry in os.listdir(show_path):
                entry_path = os.path.join(show_path, entry)
                if os.path.isdir(entry_path):
                    seasons[entry] = [f for f in os.listdir(entry_path) if is_video_file(f)]
                else:
                    if is_video_file(entry):
                        seasons["Season 1"] = seasons.get("Season 1", []) + [entry]
            shows[show] = seasons
    return shows

# Replace the search function call with the test function
# video_db = search_test()
video_db = search(video_root)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route('/shows', methods=['GET'])
def get_shows():
    return jsonify(list(video_db.keys()))

@app.route('/seasons', methods=['POST'])
def get_seasons():
    data = request.get_json()
    show_name = data.get("show_name")
    if show_name in video_db:
        return jsonify(list(video_db[show_name].keys()))
    return jsonify([])

@app.route('/episodes', methods=['POST'])
def get_episodes():
    data = request.get_json()
    show_name = data.get("show_name")
    season = data.get("season")
    if show_name in video_db and season in video_db[show_name]:
        return jsonify(video_db[show_name][season])
    return jsonify([])

@app.route('/episode_path', methods=['POST'])
def get_episode_path():
    data = request.get_json()
    show_name = data.get("show_name")
    season = data.get("season")
    episode = data.get("episode")

    if show_name in video_db and season in video_db[show_name] and episode in video_db[show_name][season]:
        episode_path = os.path.join(video_root, show_name, season, episode)
        return jsonify(episode_path)
    return jsonify("")

@app.route('/process_video', methods=['POST'])
def process_video():
    global video_queue
    data = request.get_json()
    video_path = data.get("video_path")
    print(f"video_path: {video_path}")
    print(f"exists:{os.path.exists(video_path)}")
    print(f"is_video_file:{is_video_file(video_path)}")
    if os.path.exists(video_path) and is_video_file(video_path):
        if video_path in processed_videos:
            return jsonify({"processed": True, "output_path": processed_videos[video_path]})
        else:
            print(f"Size of video_queue before adding: {video_queue.qsize()}")
            video_queue.put(video_path)
            return jsonify({"processed": False, "output_path": ""})
    return jsonify({"processed": False, "output_path": ""})

@app.route('/video_state', methods=['POST'])
def video_state():
    global processed_videos
    data = request.get_json()
    video_path = data.get("video_path")
    if os.path.exists(video_path) and is_video_file(video_path):
        if video_path in processed_videos:
            return jsonify({"processed": True, "output_path": processed_videos[video_path]})
        else:
            return jsonify({"processed": False, "output_path": ""})
    return jsonify({"processed": False, "output_path": ""})

def stop_thread():
    video_queue.put(None)
    processing_thread.join()

def load_cache_file(cache_file):
    if not os.path.exists(cache_file):
        return {}

    processed_videos = {}
    with open(cache_file, 'r') as f:
        for line in f:
            input_path, output_path = line.strip().split(',')
            processed_videos[input_path] = output_path

    # print(processed_videos)
    return processed_videos

def move_and_update_cache(input_path, processed_video, cache_folder, cache_file):
    _, filename = os.path.split(processed_video)
    output_path = os.path.join(cache_folder, filename)

    print(f"Moving processed video from {input_path} to {output_path}")
    # Move the processed video file to the cache folder
    shutil.move(processed_video, output_path)

    # Update the cache file
    with open(cache_file, 'a') as f:
        f.write(f"{input_path},{output_path}\n")
    print(f"Updated cache file with entry: {input_path},{output_path}")

    return output_path

def video_processing_thread():
    global video_queue
    print("Video processing thread started.")
    while True:
        video_path = video_queue.get()
        if video_path is None:
            break

        print(f"Processing video: {video_path}")

        if video_path in in_progress_videos:
            print(f"Video already in progress: {video_path}")
            video_queue.task_done()
            continue

        in_progress_videos.add(video_path)

        # Copy the input file to the specified folder
        input_copy_folder = "C:/Users/alec/source/python/depth_video/unformatted_videos/"
        _, filename = os.path.split(video_path)
        file_root, file_ext = os.path.splitext(filename)
        unique_filename = f"{file_root}_{uuid.uuid4().hex}{file_ext}"
        input_copy_path = os.path.join(input_copy_folder, unique_filename)
        shutil.copy(video_path, input_copy_path)
        print(f"Copied input video to: {input_copy_path}")

        print(f"Processing video: {input_copy_path}")
        output_path = depth_gen(input_copy_path, output_folder, model_path, model_type, optimize, height, square, 32)

        print(f"Deleting input copy: {input_copy_path}")
        os.remove(input_copy_path)
        
        # Move the processed video file to the cache folder and update the cache file
        cache_output_path = move_and_update_cache(video_path, output_path, cache_folder, cache_file)
        processed_videos[video_path] = cache_output_path
        print(f"Finished processing video: {video_path}")
        in_progress_videos.remove(video_path)

        video_queue.task_done()

if __name__ == "__main__":
    # Load the cache file into the processed_videos dictionary
    processed_videos = load_cache_file(cache_file)

    # Start the video processing thread
    processing_thread = threading.Thread(target=video_processing_thread)
    processing_thread.start()

    try:
        init()
    finally:
        stop_thread()