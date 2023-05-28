from flask import Flask, render_template, jsonify, request, make_response
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import struct
import cv2
import base64
import numpy as np
import time

from imageio.core.util import Array
from run import initialize_model, process_images
from pipeline import pipeline_depth_only
from output_adapter import initialize_output, write_output_video
from utils import depth_to_rgb24, batch_process, depth_to_rgb24_pixels, write_depth

app = Flask(__name__, static_folder='static', template_folder='templates')
app.debug = True


model_type = "dpt_beit_base_384"
# model_type = "dpt_beit_large_384"
# model_type = "dpt_next_vit_large_384"
# model_type = "dpt_swin2_base_384"
# model_type = "dpt_beit_large_512"

model_path = f"weights/{model_type}.pt"

# Initialize the model
model = initialize_model(model_path=model_path, model_type=model_type, optimize=True, height=None, square=False)

class Stats:
    def __init__(self):
        self.frame_count = 0
        self.total_processing_time = 0

    def add_frame(self, processing_time):
        self.frame_count += 1
        self.total_processing_time += processing_time

    def get_average_processing_time(self):
        if self.frame_count == 0:
            return 0
        return self.total_processing_time / self.frame_count

    def get_frames_per_second(self):
        if self.total_processing_time == 0:
            return 0
        return self.frame_count / (self.total_processing_time / 1000)

    def to_dict(self):
        return {
            'frame_count': self.frame_count,
            'average_processing_time_ms': self.get_average_processing_time(),
            'frames_per_second': self.get_frames_per_second(),
        }

stats = Stats()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(stats.to_dict())

is_processing = False

@app.route('/process', methods=['POST'])
def process_image():
    global is_processing

    if is_processing:
        return make_response(jsonify({'error': 'Server is busy processing another frame'}), 429)  # 429 is the HTTP status code for "Too Many Requests"

    is_processing = True
    try:
        message = request.get_data()
        frame_data = message 

        if frame_data:
            start_time = time.time()
            stream_id, width, height, timestamp, img_data = parse_frame(frame_data)
            
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            print(f"image shape: {image.shape} frame size: {width}x{height}")
            image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)
            
            input_images = [image]
            prediction = process_images(model, input_images, optimize=True, min_width=1280)

            depth_Frame = pipeline_depth_only(prediction[0], image, True)
            depth_Frame = cv2.resize(depth_Frame, (width, height), interpolation=cv2.INTER_AREA)

            _, enc_img = cv2.imencode('.jpg', depth_Frame)
            if enc_img is None:
                return jsonify({'error': 'Failed to encode image'})

            base64_img = base64.b64encode(enc_img).decode('utf-8')

            processing_time = (time.time() - start_time) * 1000
            stats.add_frame(processing_time)

            return jsonify({'stream_id': stream_id, 'width': width, 'height': height, 'timestamp': timestamp, 'image': base64_img})
    except Exception as e:
        print(f'Error processing image: {e}')
        return make_response(jsonify({'error': 'Error processing image'}), 500)  # 500 is the HTTP status code for "Internal Server Error"
    finally:
        is_processing = False

def parse_frame(data):
    stream_id = data[0]
    width = struct.unpack('H', data[1:3])[0]
    height = struct.unpack('H', data[3:5])[0]
    timestamp = struct.unpack('q', data[5:13])[0]
    img_data = data[13:]
    return stream_id, width, height, timestamp, img_data

def construct_frame(stream_id, width, height, timestamp, img_data):
    header = bytearray(13)
    header[0] = stream_id
    header[1:3] = struct.pack('H', width)
    header[3:5] = struct.pack('H', height)
    header[5:13] = struct.pack('q', timestamp)
    return header + img_data

if __name__ == '__main__':
    batch_size = 1
    app.run(host='0.0.0.0', port=5000)
