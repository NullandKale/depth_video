import os
import generate_depth_videos

def is_video_file(filename):
    supported_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.wedbm', '.m4v', '.3gp']
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in supported_extensions

if __name__ == "__main__":
    output_folder = "output/"

    # model_type = "dpt_beit_base_384"
    model_type = "dpt_beit_large_384"
    # model_type = "dpt_next_vit_large_384"
    # model_type = "dpt_swin2_base_384"
    # model_type = "dpt_beit_large_512"

    model_path = f"weights/{model_type}.pt"
    optimize = True
    height = None
    square = False
    batch_size = 1

    input_folder = "unformatted_videos/"

    # Register the signal handler
    generate_depth_videos.signal.signal(generate_depth_videos.signal.SIGINT, generate_depth_videos.signal_handler)

    for filename in os.listdir(input_folder):
        if is_video_file(filename):
            video_path = os.path.join(input_folder, filename)
            print(f"Processing video file: {video_path}")
            generate_depth_videos.process_video(video_path, output_folder, model_path, model_type, optimize, height, square, batch_size)
