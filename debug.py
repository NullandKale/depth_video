import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_average(image, edge_percentage, center_percentage, threshold):
    edge_region_pixels = edge_region(image, edge_percentage)
    center_average_color = center_average(image, center_percentage)

    # Exclude values around the center_average_color using the threshold
    mask = np.abs(edge_region_pixels - center_average_color) > threshold
    filtered_edge_region_pixels = edge_region_pixels[mask]

    return np.mean(filtered_edge_region_pixels)

def edge_region(image, edge_percentage):
    height, width = image.shape
    edge_width = int(edge_percentage * width)
    edge_height = int(edge_percentage * height)

    top = image[:edge_height, :]
    bottom = image[-edge_height:, :]
    left = image[edge_height:-edge_height, :edge_width]
    right = image[edge_height:-edge_height, -edge_width:]

    top_flat = top.flatten()
    bottom_flat = bottom.flatten()
    left_flat = left.flatten()
    right_flat = right.flatten()

    combined = np.concatenate((top_flat, bottom_flat, left_flat, right_flat))

    # Reshape the combined array into a 2D array
    reshaped = combined.reshape(-1, 1)

    return reshaped

def center_average(image, center_percentage):
    center_width = int(min(image.shape[0], image.shape[1]) * center_percentage)
    h, w = image.shape[:2]
    top_left = (h // 2 - center_width // 2, w // 2 - center_width // 2)
    center_area = image[top_left[0]:top_left[0] + center_width, top_left[1]:top_left[1] + center_width]

    return np.mean(center_area)

def get_edge_and_center_shapes(image, edge_perc, center_perc):
    h, w = image.shape
    edge_w, edge_h = int(edge_perc * w), int(edge_perc * h)
    center_w = int(min(h, w) * center_perc)

    edge_shapes = [(0, 0, edge_h, w), (h - edge_h, 0, edge_h, w), (edge_h, 0, h - 2 * edge_h, edge_w), (edge_h, w - edge_w, h - 2 * edge_h, edge_w)]
    center_shape = (h // 2 - center_w // 2, w // 2 - center_w // 2, center_w, center_w)
    return edge_shapes, center_shape

def tint_pixels_around(grey_scale_image, value, threshold, color):
    img_color = cv2.cvtColor(grey_scale_image, cv2.COLOR_GRAY2BGR)
    mask = np.abs(grey_scale_image - value) <= threshold

    for c in range(3):
        img_color[:, :, c] = np.where(mask, img_color[:, :, c] + color[c], img_color[:, :, c])

    return img_color

def tint_most_like_center_and_different_from_edge(grey_scale_image, center_avg, edge_avg, threshold, color):
    img_color = cv2.cvtColor(grey_scale_image, cv2.COLOR_GRAY2BGR)

    center_diff = np.abs(grey_scale_image - center_avg)
    edge_diff = np.abs(grey_scale_image - edge_avg)

    # Identify the regions most like the center and least like the edge
    desired_mask = np.logical_and(center_diff < threshold, edge_diff > threshold)

    for c in range(3):
        img_color[:, :, c] = np.where(desired_mask, img_color[:, :, c] + color[c], img_color[:, :, c])

    return img_color

def show_image_parts(image_path, edge_perc=0.1, center_perc=0.3, threshold=20):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    center_avg = center_average(img, center_perc)
    edge_avg = edge_average(img, edge_perc, center_perc, 10)
    most_like_center_and_different_from_edge = tint_most_like_center_and_different_from_edge(img, center_avg, edge_avg, threshold, (0, 255, 0))

    plt.figure()
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap='gray'), plt.title('Original')
    plt.subplot(122), plt.imshow(cv2.cvtColor(most_like_center_and_different_from_edge, cv2.COLOR_BGR2RGB)), plt.title('Most Like Center & Different from Edge (Green)')
    plt.show()

if __name__ == '__main__':
    img_path = './out/depth_frames/frame_0000384-dpt_beit_large_512.png'
    if os.path.isfile(img_path):
        show_image_parts(img_path)
    else:
        print(f"ERROR: File '{img_path}' not found.")