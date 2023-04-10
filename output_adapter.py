import os
import subprocess

class VideoWriter:
    def __init__(self, writer, output_video_filename):
        self.writer = writer
        self.output_video_filename = output_video_filename

    def write(self, combined_frame):
        if combined_frame is not None:
            self.writer.stdin.write(combined_frame.tobytes())

    def close(self):
        self.writer.stdin.close()
        self.writer.wait()

def initialize_output(output_folder, frame_size, fps, video_path, output_mode='file', rtsp_server='127.0.0.1', rtsp_port='8554', pipe_output=False):
    if output_mode == 'file':
        return initialize_file_output(output_folder, frame_size, fps, video_path, pipe_output)
    elif output_mode == 'rtsp':
        return initialize_rtsp_output(frame_size, fps, rtsp_server, rtsp_port, pipe_output)
    else:
        raise ValueError("Invalid output mode specified")

def initialize_rtsp_output(frame_size, fps, rtsp_server, rtsp_port, pipe_output):
    output_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-maxrate', '8M', '-bufsize', '512M']
    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{frame_size[0]}x{frame_size[1]}", '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-f', 'rtsp', '-muxdelay', '0', *output_args, f"rtsp://{rtsp_server}:{rtsp_port}/stream.sdp"]
    if pipe_output:
        video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    else:
        video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return VideoWriter(video_writer, f"rtsp://{rtsp_server}:{rtsp_port}/stream.sdp")

def initialize_file_output(output_folder, frame_size, fps, video_path, pipe_output):
    os.makedirs(output_folder, exist_ok=True)
    video_filename = os.path.basename(video_path)
    output_video_filename = os.path.join(output_folder, video_filename.replace(os.path.splitext(video_filename)[1], '_output.mp4'))
    output_args = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-maxrate', '8M', '-bufsize', '512M']
    ffmpeg_cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f"{frame_size[0]}x{frame_size[1]}", '-r', str(fps), '-i', '-', '-i', video_path, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-movflags', 'faststart', '-muxdelay', '0', *output_args, output_video_filename]
    if pipe_output:
        video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    else:
        video_writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return VideoWriter(video_writer, output_video_filename)

def write_output_video(combined_frame, video_writer):
    """Write combined frames to output video."""
    if combined_frame is not None:
        video_writer.write(combined_frame)