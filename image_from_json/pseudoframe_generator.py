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
os.makedirs("./heat_pseudoframes_images", exist_ok=True)

# Iterate over each sample
for timestamp, sample_values in decoded_samples.items():
    counter += 1
    print(f"Processing timestamp: {timestamp} ({counter}/{total})")

    # Converting the list of temperatures into a 32x24 array
    temperature_array = np.array(sample_values).reshape(24, 32)

    # Get the pages from the frame (the intercalated pixels)
    page_0, page_1 = utils.get_pages_from_frame(temperature_array)

    # Create the arrays for the pseudoframes
    pseudoframe_0 = np.zeros((24, 32))
    pseudoframe_1 = np.zeros((24, 32))

    # Iterate over the rows and collumns of the frame, filling the pseudoframes
    for row in range(24):
        for col in range(32):
            
            if (row + col) % 2 == 0: # Fill the pseudoframe 0
                pseudoframe_0[row][col] = page_0[row][col // 2]
            else: # Fill the pseudoframe 1
                pseudoframe_1[row][col] = page_1[row][col // 2]

    # Interpolate the black pixels according to the surrounding pixels
    pseudoframe_0 = utils.interpolate_black_pixels(pseudoframe_0)
    pseudoframe_1 = utils.interpolate_black_pixels(pseudoframe_1)

    # Evaluate the mean of the pseudoframes
    pseudoframe_mean = (pseudoframe_0 + pseudoframe_1) / 2

    # Evaluate the absolute difference between the pseudoframes
    pseudoframe_diff = np.abs(pseudoframe_0 - pseudoframe_1)

    pseudoframe_mean_plus_diff = pseudoframe_mean + pseudoframe_diff / 2
    pseudoframe_mean_minus_diff = pseudoframe_mean - pseudoframe_diff / 2

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Set the figure suptitle
    fig.suptitle(f"Pseudoframes analysis: {timestamp} ({utils.format_time(timestamp)})", fontsize=16)

    # Plot the original frame
    heatmap_full = axs[0][0].imshow(temperature_array, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_full, ax=axs[0][0], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[0][0].set_title("Full Frame")

    # Plot the pseudoframe 0
    heatmap_pseudoframe_0 = axs[0][1].imshow(pseudoframe_0, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_pseudoframe_0, ax=axs[0][1], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[0][1].set_title("Pseudoframe 0")

    # Plot the pseudoframe 1
    heatmap_pseudoframe_1 = axs[0][2].imshow(pseudoframe_1, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_pseudoframe_1, ax=axs[0][2], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[0][2].set_title("Pseudoframe 1")

    # Plot the pseudoframe mean
    heatmap_pseudoframe_mean = axs[1][0].imshow(pseudoframe_mean, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_pseudoframe_mean, ax=axs[1][0], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[1][0].set_title("Pseudoframe Mean")

    # Plot the pseudoframe plus diff
    heatmap_pseudoframe_mean_plus_diff = axs[1][1].imshow(pseudoframe_mean_plus_diff, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_pseudoframe_mean_plus_diff, ax=axs[1][1], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[1][1].set_title("Pseudoframe Mean + Diff")

    # Plot the pseudoframe minus diff
    heatmap_pseudoframe_mean_minus_diff = axs[1][2].imshow(pseudoframe_mean_minus_diff, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_pseudoframe_mean_minus_diff, ax=axs[1][2], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[1][2].set_title("Pseudoframe Mean - Diff")

    # Adjust layout
    plt.tight_layout()

    # Saving the figure
    plt.savefig(f"./heat_pseudoframes_images/heat_pseudoframes_image_{timestamp}.png")
    plt.close(fig)

    # break # DEBUG




