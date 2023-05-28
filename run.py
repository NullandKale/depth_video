"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
import time
import logging
import warnings
import numpy as np

from imutils.video import VideoStream
from midas.model_loader import default_models, load_model

first_execution = True
saved_output = False

warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

class MidasModel:
    def __init__(self, device, model, model_type, transform):
        self.device = device
        self.model = model
        self.model_type = model_type
        self.transform = transform

def resize_frame(frame, min_width=384):
    height, width, _ = frame.shape
    if width >= min_width:
        return frame.copy()

    new_width = min_width
    aspect_ratio = float(height) / float(width)
    new_height = int(new_width * aspect_ratio)

    return cv2.resize(frame.copy(), (new_width, new_height), interpolation=cv2.INTER_AREA)

def scale_depth_values(predictions, bits=2):
    max_val = (2**(8*bits))-1
    scaled_predictions = []

    for prediction in predictions:
        depth_min, depth_max = np.nanmin(prediction), np.nanmax(prediction)
        range_diff = depth_max - depth_min

        if np.isfinite(range_diff) and range_diff > np.finfo("float").eps:
            out = max_val * (prediction - depth_min) / range_diff
        else:
            out = np.zeros(prediction.shape, dtype=prediction.dtype)

        scaled_predictions.append(out)

    return scaled_predictions

def resize_frame_torch(frame, min_width=384):
    height, width, _ = frame.shape
    if width >= min_width:
        return torch.from_numpy(frame)

    new_width = min_width
    aspect_ratio = float(height) / float(width)
    new_height = int(new_width * aspect_ratio)

    frame_torch = torch.from_numpy(frame)
    frame_torch = frame_torch.permute(2, 0, 1).unsqueeze(0)
    resized_frame_torch = torch.nn.functional.interpolate(
        frame_torch, (new_height, new_width), mode='bilinear', align_corners=False
    )
    return resized_frame_torch.squeeze(0).permute(1, 2, 0)

def initialize_model(model_path, model_type="dpt_beit_large_512", optimize=False, height=None, square=False):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, transform, _, _ = load_model(device, model_path, model_type, optimize, height, square)
    except Exception as e:
        print("Error during model loading:", e)
        return None
    return MidasModel(device, model, model_type, transform)

def process_images(midas_model, batch_images, optimize=False, min_width=384):
    # print(f"type(frames): {type(batch_images)}, len(frames): {len(batch_images)}")
    # print(f"frames[0]: type={type(batch_images[0])}, shape={batch_images[0].shape}, dtype={batch_images[0].dtype}")
    # exit()
    batch_original_images_rgb = batch_images
    batch_transformed_images = []
    for original_image_rgb in batch_original_images_rgb:
        # Normalize the original image's pixel values
        original_image_rgb = original_image_rgb.astype(np.float32) / 255.0
        transformed_image = midas_model.transform({"image": original_image_rgb})["image"]
        batch_transformed_images.append(transformed_image)

    with torch.no_grad():
        batch_predictions = process(midas_model, batch_transformed_images, [(image.shape[0], image.shape[1]) for image in batch_original_images_rgb], optimize)

    # Scale the depth predictions
    batch_predictions = scale_depth_values(batch_predictions)

    return batch_predictions

def process(midas_model, images, target_sizes, optimize):
    device = midas_model.device
    model = midas_model.model

    sample = torch.stack([torch.from_numpy(image) for image in images]).to(device)

    if optimize and device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    predictions = model.forward(sample)

    # Interpolate each prediction to match the original input size
    interpolated_predictions = [
        torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(1),
            size=(target_size[0], target_size[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        for pred, target_size in zip(predictions, target_sizes)
    ]

    # Convert the interpolated predictions back to numpy array
    predictions = [pred.cpu().numpy() for pred in interpolated_predictions]

    return predictions


def run(input_path, output_path, model_path, model_type, optimize, side, height, square, grayscale):
    # initialize the model
    model  = initialize_model(model_path, model_type, optimize, height, square)
    if model is None:
        return

    # get input
    if input_path is not None:
        image_names = glob.glob(os.path.join(input_path, "*"))
        num_images = len(image_names)

    # create output folder
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    # Modify batch_size according to your requirements
    batch_size = 16

    # Loop through the images in batches
    for index in range(0, num_images, batch_size):
        batch_image_names = image_names[index : index + batch_size]

        # Read and preprocess images in the batch
        batch_original_images_rgb = [utils.read_image(image_name) for image_name in batch_image_names]

        # Compute depth maps for the batch
        batch_predictions = process_images(model, batch_original_images_rgb, optimize)

        # Save the depth maps
        for i, prediction in enumerate(batch_predictions):
            image_name = batch_image_names[i]
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
                )
                utils.write_depth(filename, prediction, grayscale, bits=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', default=None)
    parser.add_argument('-o', '--output_path', default=None)
    parser.add_argument('-m', '--model_weights', default=None)
    parser.add_argument('-t', '--model_type', default='dpt_beit_large_512')
    parser.add_argument('-s', '--side', action='store_true')
    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.set_defaults(optimize=False)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--square', action='store_true')
    parser.add_argument('--grayscale', action='store_true')

    args = parser.parse_args()
    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, True)
