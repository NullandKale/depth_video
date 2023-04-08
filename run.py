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

def initialize_model(model_path, model_type="dpt_beit_large_512", optimize=False, height=None, square=False):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    except Exception as e:
        print("Error during model loading:", e)
        return None, None, None, None, None
    return device, model, transform, net_w, net_h

def run(input_path, output_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=True):
    # initialize the model
    device, model, transform, net_w, net_h = initialize_model(model_path, model_type, optimize, height, square)
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

        # cv2.imwrite("out/test_input_frame_run.png", cv2.cvtColor(batch_original_images_rgb[0], cv2.COLOR_RGB2BGR))

        # Compute depth maps for the batch
        batch_predictions = process_images(device, model, model_type, transform, net_w, net_h, batch_original_images_rgb, optimize)

        # Save the depth maps
        for i, prediction in enumerate(batch_predictions):
            image_name = batch_image_names[i]
            if output_path is not None:
                filename = os.path.join(
                    output_path, os.path.splitext(os.path.basename(image_name))[0] + '-' + model_type
                )
                utils.write_depth(filename, prediction, grayscale, bits=2)

def process_images(device, model, model_type, transform, net_w, net_h, batch_images, optimize=False, prefix=""):
    try:
        batch_original_images_rgb = batch_images
        batch_transformed_images = [transform({"image": original_image_rgb})["image"] for original_image_rgb in batch_original_images_rgb]

        # print(f"{prefix}Input image shape: {batch_original_images_rgb[0].shape}, data type: {batch_original_images_rgb[0].dtype}")

        with torch.no_grad():
            batch_predictions = process(device, model, model_type, batch_transformed_images, (net_w, net_h), [(image.shape[0], image.shape[1]) for image in batch_original_images_rgb], optimize)

        global saved_output
        if not saved_output:
            # Save a single test image
            saved_output = True
            utils.write_depth("out/test_process_images.png", batch_predictions[0], True, bits=2)

        # Scale the depth predictions
        batch_predictions = utils.scale_depth_values(batch_predictions)

        return batch_predictions

    except Exception as e:
        print(f"{prefix}Error during image processing:", e)
        return None

def process(device, model, model_type, images, input_size, target_sizes, optimize):
    global first_execution

    sample = torch.stack([torch.from_numpy(image) for image in images]).to(device)

    if optimize and device == torch.device("cuda"):
        if first_execution:
            print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                    "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                    "  half-floats.")
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    predictions = model.forward(sample)

    # Calculate scale factors for each image in the batch
    scale_factors = [target_size[-1] / predictions.shape[-1] for target_size in target_sizes]

    # Interpolate each prediction with the corresponding scale factor
    interpolated_predictions = [
        torch.nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(1),
            size=None,
            scale_factor=scale_factor,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        for pred, scale_factor in zip(predictions, scale_factors)
    ]

    # Convert the interpolated predictions back to numpy array
    predictions = [pred.cpu().numpy() for pred in interpolated_predictions]

    return predictions

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
