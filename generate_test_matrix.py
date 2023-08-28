import os
import generate_depth_videos
from pipeline import generate_settings_range, get_setting_name_by_index, get_settings_count, get_min, get_max, get_step

def is_video_file(filename):
    supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp']
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in supported_extensions

def process_video_file(video_path, min_settings, max_settings, steps_settings):
    print(f"Processing video file: {video_path}")

    # Iterate through all the attributes in the PipelineSettings class
    for i in range(get_settings_count()):
        attr_name = get_setting_name_by_index(i)
        min_value = getattr(min_settings, attr_name)
        max_value = getattr(max_settings, attr_name)

        # Print the attribute name and value
        print(f"Attribute: {attr_name}")
        print(f"Steps value: {getattr(steps_settings, attr_name)}")

        steps = getattr(steps_settings, attr_name)

        # Only process if steps is greater than 0
        if steps > 0:
            settings_range = generate_settings_range(attr_name, min_value, max_value, steps)
            setting_folder = os.path.join(output_folder, attr_name)

            if not os.path.exists(setting_folder):
                os.makedirs(setting_folder)

            for setting in settings_range:
                generate_depth_videos.process_video(video_path, setting_folder, model_path, model_type, optimize, height, square, batch_size, settings=setting)

if __name__ == "__main__":
    output_folder = "output/"
    model_type = "dpt_beit_large_384"
    model_path = f"weights/{model_type}.pt"
    optimize = True
    height = 384
    square = False
    batch_size = 24
    input_folder = "unformatted_videos/"

    # Define min, max, and steps for the range of settings
    min_settings = get_min()
    max_settings = get_max()
    steps_settings = get_step()

    # Register the signal handler
    generate_depth_videos.signal.signal(generate_depth_videos.signal.SIGINT, generate_depth_videos.signal_handler)

    for filename in os.listdir(input_folder):
        if is_video_file(filename):
            video_path = os.path.join(input_folder, filename)
            process_video_file(video_path, min_settings, max_settings, steps_settings)
