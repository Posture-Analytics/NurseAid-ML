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
os.makedirs("./heat_page_images", exist_ok=True)

# Iterate over each sample
for timestamp, sample_values in decoded_samples.items():
    counter += 1
    print(f"Processing timestamp: {timestamp} ({counter}/{total})")

    # Converting the list of temperatures into a 32x24 array
    temperature_array = np.array(sample_values).reshape(24, 32)

    # Define the range of temperatures you want to display
    min_temp = temperature_array.min()
    max_temp = temperature_array.max()

    # Get the pages from the frame (the intercalated pixels)
    page_0, page_1 = utils.get_pages_from_frame(temperature_array)

    # Calculate the absolute difference between the pages
    page_diff = np.abs(page_0 - page_1)

    # Caculate the mean of the pages
    page_mean = (page_0 + page_1) / 2

    # Calculate the sum minus the absolute difference over 2
    page_sum = (page_0 + page_1 - page_diff) / 2

    # Evaluate the sum and the maximum difference between the pages
    sum_diff.append(page_diff.sum())
    max_diff.append(page_diff.max())

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(6, 1, figsize=(12, 36))
    
    # Set the figure suptitle
    fig.suptitle(f"Movement betwwen pages analysis: {timestamp} ({utils.format_time(timestamp)})", fontsize=16)

    # Plotting the full frame
    heatmap_full = axs[0].imshow(temperature_array, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_full, ax=axs[2], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[2].set_title("Full Frame")

    # Plotting the heat image for even indexes
    heatmap_even = axs[1].imshow(page_0, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_even, ax=axs[0], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[0].set_title("Page 0")

    # Plotting the heat image for odd indexes
    heatmap_odd = axs[2].imshow(page_1, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_odd, ax=axs[1], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[1].set_title("Page 1")

    # Plotting the absolute difference
    heatmap_diff = axs[3].imshow(page_diff, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_diff, ax=axs[3], label='Temperature Difference (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[3].set_title("Absolute Difference Between Pages")

    # Plotting the mean of the pages
    heatmap_mean = axs[4].imshow(page_mean, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_mean, ax=axs[4], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[4].set_title("Mean of the Pages")

    # Plotting the sum minus the absolute difference over 2
    heatmap_sum = axs[5].imshow(page_sum, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap_sum, ax=axs[5], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[5].set_title("Sum minus the absolute difference over 2")

    # Adjust layout
    plt.tight_layout()

    # Saving the figure
    plt.savefig(f"./heat_page_images/heat_page_image_{timestamp}.png")
    plt.close(fig)

    # break # DEBUG

# Create a figure with 2 subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Set the figure suptitle
fig.suptitle(f"Movement betwwen pages analysis", fontsize=16)

# Plotting the sum of the pages
axs[0].plot(sum_diff)
axs[0].set_title("Sum of Difference Between Pages")
axs[0].set_xlabel("Frame Index")
axs[0].set_ylabel("Sum")

# Plotting the maximum difference between the pages
axs[1].plot(max_diff)
axs[1].set_title("Maximum Difference Between Pages")
axs[1].set_xlabel("Frame Index")
axs[1].set_ylabel("Maximum Difference")

# Adjust layout
plt.tight_layout()

# Saving the figure
plt.savefig(f"./heat_page_image_sum_max.png")