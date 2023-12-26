import cv2
import json
import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import utils

decoded_samples = utils.load_samples('samples.json')

counter = 0
total = len(decoded_samples)

sum_diff = []
max_diff = []

# Ensure the directory for saving images exists
os.makedirs("./filtered_images", exist_ok=True)

# Iterate over each sample
for timestamp, sample_values in decoded_samples.items():
    counter += 1
    print(f"Processing timestamp: {timestamp} ({counter}/{total})")

    # Converting the list of temperatures into a 32x24 array
    temperature_array = np.array(sample_values).reshape(24, 32)

    # Apply the gaussian filter to the frame
    gaussian_3x3 = cv2.GaussianBlur(temperature_array, (3, 3), 0)
    gaussian_5x5 = cv2.GaussianBlur(temperature_array, (5, 5), 0)
    gaussian_7x7 = cv2.GaussianBlur(temperature_array, (7, 7), 0)

    # Apply the median filter to the frame
    min_val, max_val = np.min(temperature_array), np.max(temperature_array)

    # Normalize the array to 0-255
    normalized_array = 255 * (temperature_array - min_val) / (max_val - min_val)
    normalized_array = normalized_array.astype(np.uint8)

    median_3x3 = cv2.medianBlur(normalized_array, 3)
    median_5x5 = cv2.medianBlur(normalized_array, 5)
    median_7x7 = cv2.medianBlur(normalized_array, 7)

    # Apply the bilateral filter to the frame
    bilateral_3x3 = cv2.bilateralFilter(normalized_array, 3, 75, 75)
    bilateral_5x5 = cv2.bilateralFilter(normalized_array, 5, 75, 75)
    bilateral_7x7 = cv2.bilateralFilter(normalized_array, 7, 75, 75)


    # Create a figure with 12 subplots
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))

    # Set the figure suptitle
    fig.suptitle(f"Filters analysis: {timestamp}", fontsize=16)

    # Plot the original frame
    axs[0][1].imshow(temperature_array, cmap='gray')
    axs[0][1].set_title("Original frame")
    
    # Plot the gaussian 3x3 filtered frame
    axs[1][0].imshow(gaussian_3x3, cmap='gray')
    axs[1][0].set_title("Gaussian 3x3 filtered frame")

    # Plot the gaussian 5x5 filtered frame
    axs[1][1].imshow(gaussian_5x5, cmap='gray')
    axs[1][1].set_title("Gaussian 5x5 filtered frame")

    # Plot the gaussian 7x7 filtered frame
    axs[1][2].imshow(gaussian_7x7, cmap='gray')
    axs[1][2].set_title("Gaussian 7x7 filtered frame")

    # Plot the median 3x3 filtered frame
    axs[2][0].imshow(median_3x3, cmap='gray')
    axs[2][0].set_title("Median 3x3 filtered frame")

    # Plot the median 5x5 filtered frame
    axs[2][1].imshow(median_5x5, cmap='gray')
    axs[2][1].set_title("Median 5x5 filtered frame")

    # Plot the median 7x7 filtered frame
    axs[2][2].imshow(median_7x7, cmap='gray')
    axs[2][2].set_title("Median 7x7 filtered frame")

    # Plot the bilateral 3x3 filtered frame
    axs[3][0].imshow(bilateral_3x3, cmap='gray')
    axs[3][0].set_title("Bilateral 3x3 filtered frame")

    # Plot the bilateral 5x5 filtered frame
    axs[3][1].imshow(bilateral_5x5, cmap='gray')
    axs[3][1].set_title("Bilateral 5x5 filtered frame")

    # Plot the bilateral 7x7 filtered frame
    axs[3][2].imshow(bilateral_7x7, cmap='gray')
    axs[3][2].set_title("Bilateral 7x7 filtered frame")

    # Delete the extra subplots
    fig.delaxes(axs[0][0])
    fig.delaxes(axs[0][2])

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"./filtered_images/{timestamp}.png")
    plt.close(fig)

    # break #DEBUG