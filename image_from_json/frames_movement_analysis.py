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

# Convert the dictionary into a list of tuples
sample_list = list(decoded_samples.items())

# Iterate over each sample
for i in range(len(sample_list) - 1):

    # Get the current and next samples
    current_timestamp, current_sample_values = sample_list[i]
    next_timestamp, next_sample_values = sample_list[i + 1]

    counter += 1
    print(f"Processing timestamps: {current_timestamp} - {next_timestamp} ({counter}/{total})") 

    # Evaluate the time difference between the samples
    time_difference = int(next_timestamp) - int(current_timestamp)

    # Converting the list of temperatures into a 32x24 array
    current_temperature_array = np.array(current_sample_values).reshape(24, 32)
    next_temperature_array = np.array(next_sample_values).reshape(24, 32)

    # Calculate the absolute difference between the frames
    frame_diff = np.abs(current_temperature_array - next_temperature_array)

    # Caculate the mean of the frames
    frame_mean = (current_temperature_array + next_temperature_array) / 2

    # Calculate the sum minus the absolute difference over 2
    frame_sum = (current_temperature_array + next_temperature_array - frame_diff) / 2

    # Evaluate the sum and the maximum difference between the frames
    sum_diff.append(frame_diff.sum())
    max_diff.append(frame_diff.max())

    # # Create a figure with 5 subplots
    # fig, axs = plt.subplots(5, 1, figsize=(12, 30))
    
    # # Set the figure suptitle
    # fig.suptitle(f"Movement betwwen frames analysis: {current_timestamp} - {next_timestamp}", fontsize=16)

    # # Plotting the current frame
    # heatmap_current = axs[0].imshow(current_temperature_array, cmap='gray', interpolation='nearest')
    # fig.colorbar(heatmap_current, ax=axs[0], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    # axs[0].set_title("Current Frame")

    # # Plotting the next frame
    # heatmap_next = axs[1].imshow(next_temperature_array, cmap='gray', interpolation='nearest')
    # fig.colorbar(heatmap_next, ax=axs[1], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    # axs[1].set_title("Next Frame")

    # # Plotting the absolute difference
    # heatmap_diff = axs[2].imshow(frame_diff, cmap='gray', interpolation='nearest')
    # fig.colorbar(heatmap_diff, ax=axs[2], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    # axs[2].set_title("Absolute Difference")

    # # Plotting the mean
    # heatmap_mean = axs[3].imshow(frame_mean, cmap='gray', interpolation='nearest')
    # fig.colorbar(heatmap_mean, ax=axs[3], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    # axs[3].set_title("Mean")

    # # Plotting the sum minus the absolute difference over 2
    # heatmap_sum = axs[4].imshow(frame_sum, cmap='gray', interpolation='nearest')
    # fig.colorbar(heatmap_sum, ax=axs[4], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    # axs[4].set_title("Sum minus the absolute difference over 2")

    # # Adjust layout
    # plt.tight_layout()

    # # Save the figure
    # plt.savefig(f"./heat_frame_images/{current_timestamp}_{next_timestamp}.png")
    # plt.close(fig)

    # break #DEBUG

# Create a figure with 2 subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Set the figure suptitle
fig.suptitle(f"Movement betwwen frames analysis", fontsize=16)

# Plotting the sum difference
axs[0].plot(sum_diff)
axs[0].set_title("Sum of Difference Between Frames")
axs[0].set_xlabel("Frame Index")
axs[0].set_ylabel("Sum Difference")

# Plotting the maximum difference
axs[1].plot(max_diff)
axs[1].set_title("Maximum Difference Between Frames")
axs[1].set_xlabel("Frame Index")
axs[1].set_ylabel("Maximum Difference")

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(f"./heat_frame_images_sum_max_diff.png")
plt.close(fig)