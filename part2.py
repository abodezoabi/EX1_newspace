import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import filters


def detect_and_log(path, index):
    img = imread(path)
    if img is None:
        print('Error: Failed to load image', path)
        return False

    # Remove alpha channel if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    gray = rgb2gray(img)
    blurred = filters.gaussian(gray, sigma=1)

    blobs = blob_log(
        blurred,
        min_sigma=3,
        max_sigma=6,
        num_sigma=10,
        threshold=0.05
    )

    # Filter based on brightness and size
    filtered_blobs = []
    for blob in blobs:
        y, x, r = blob
        patch = gray[int(y) - 2:int(y) + 2, int(x) - 2:int(x) + 2]
        brightness = np.mean(patch)
        if brightness > 0.2 and r > 1.5:
            filtered_blobs.append((y, x, r, brightness))

    # Logging
    log_filename = f"stars_output_image_{index}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"Processing: {path}\n")
        log_file.write(f"Number of stars in picture {index}: {len(filtered_blobs)}\n")

        print(f"Detecting image path: {path}")
        print(f"Number of stars in picture {index}: {len(filtered_blobs)}")

        fig, ax = plt.subplots()
        ax.imshow(img)

        for i, (y, x, r, brightness) in enumerate(filtered_blobs, start=1):
            circ = plt.Circle((x, y), r * 1.5, color='red', linewidth=1.5, fill=False)
            ax.add_patch(circ)

            output = f"Star {i}: x={x:.1f}, y={y:.1f}, r={r:.1f}, brightness={brightness:.3f}"
            print(output)
            log_file.write(output + "\n")

        plt.title(f"Filtered Stars in Image {index}")
        plt.axis('off')
        plt.show()

    print(f"Log saved to: {log_filename}\n")
    return True


if __name__ == '__main__':
    image_folder = os.path.join(os.getcwd(), "img")
    image_paths = []

    if not os.path.exists(image_folder):
        print(f"Image folder not found at: {image_folder}")
    else:
        for dirpath, subdirs, filenames in os.walk(image_folder):
            for file in filenames:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(dirpath, file))

        print("Number of images found:", len(image_paths))
        for index, path in enumerate(image_paths, start=1):
            print("Processing:", path)
            detect_and_log(path=path, index=index)
