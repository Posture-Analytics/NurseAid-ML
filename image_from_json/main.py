import cv2
import json
import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import base64_thermal_decoder as decoder

HISTOGRAM_RESOLUTION = 0.5 # 0.1°C

def format_time(timestamp):

    # Convert the string of a millisecond timestamp into a int of a second timestamp
    timestamp = int(timestamp) / 1000

    # Convert the timestamp into a datetime object
    timestamp = datetime.fromtimestamp(timestamp)

    # Convert the datetime object into a formatted string
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Chop off the last 3 digits of the string (the microseconds) and add the timezone
    return time_str[:-3] + " GMT-3"

# Read the json file with the samples into a dictionary
with open('samples.json', 'r') as f:
    samples = json.load(f)

# Ensure the directory for saving images exists
os.makedirs("./heat_images", exist_ok=True)

decoded_samples = {}

# Iterate over each sample
for timestamp, encoded_string in samples.items():
    # Decode the string
    decoded_values = decoder.decode_base64(encoded_string, 0, 100.0)

    # Assign the decoded values to the dictionary
    decoded_samples[timestamp] = decoded_values

# Get a random sample
timestamp = random.choice(list(decoded_samples.keys()))
sample_values = decoded_samples.get(timestamp)

counter = 1
total = len(decoded_samples)

# Iterate over each sample
for timestamp, sample_values in decoded_samples.items():
    print(f"Processing timestamp: {timestamp} ({counter}/{total})")

    # Converting the list of temperatures into a 32x24 array
    temperature_array = np.array(sample_values).reshape(24, 32)

    # Define the range of temperatures you want to display
    min_temp = temperature_array.min()
    max_temp = temperature_array.max()

    # Calculate summary statistics
    mean = np.mean(temperature_array)
    median = np.median(temperature_array)
    percentile_25 = np.percentile(temperature_array, 25)
    percentile_75 = np.percentile(temperature_array, 75)

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Set the figure suptitle
    fig.suptitle(f"Thermal Image Analysis: {timestamp} ({format_time(timestamp)})", fontsize=16)

    # Plotting the heat image
    heatmap = axs[0].imshow(temperature_array, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap, ax=axs[0], label='Temperature (°C)', orientation='horizontal', fraction=0.05, pad=0.05)
    axs[0].set_title(f"Heat Image")

    # Evaluating the bins of the histogram
    bin_edges = np.arange(math.floor(min_temp), math.ceil(max_temp) + HISTOGRAM_RESOLUTION, HISTOGRAM_RESOLUTION)

    # Plotting the frequency distribution
    temperature_values = temperature_array.flatten()
    n, bins, patches = axs[1].hist(temperature_values, bins=bin_edges, cumulative=False, density=True, alpha=1, rwidth=1, label='Frequency', edgecolor='black', linewidth=1)

    # Mapping the colors from the heatmap
    colormap = plt.cm.gray
    norm = plt.Normalize(min_temp, max_temp)
    for bin, patch in zip(bins, patches):
        color = colormap(norm(bin))
        patch.set_facecolor(color)

    # Adding summary statistics lines with corresponding colors
    axs[1].axvline(mean, color=colormap(norm(mean)), linestyle='dashed', linewidth=3, label=f'Mean: {mean:.2f}°C')
    axs[1].axvline(percentile_25, color=colormap(norm(percentile_25)), linestyle='dotted', linewidth=3, label=f'25th Percentile: {percentile_25:.2f}°C')
    axs[1].axvline(median, color=colormap(norm(median)), linestyle='dotted', linewidth=3, label=f'50th Percentile (Median): {median:.2f}°C')
    axs[1].axvline(percentile_75, color=colormap(norm(percentile_75)), linestyle='dotted', linewidth=3, label=f'75th Percentile: {percentile_75:.2f}°C')
    axs[1].set_title("Frequency Distribution")
    axs[1].set_xlabel("Temperature (°C)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xticks(np.arange(int(min_temp), int(max_temp) + 1, 1))
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Saving the figure
    plt.savefig(f"./heat_images/heat_image_{timestamp}.png")
    plt.close(fig)



# Open all images and convert to a video (1 image per second) using cv2
    
print("Creating video from images...")
    
# Directory where the images are stored
image_directory = './heat_images'

# Output video file
output_video_file = 'thermal_video.avi'

# Video properties
frame_rate = 1  # 1 frame per second

# Initialize video writer
video_writer = None

# Iterate over each image
for timestamp, _ in decoded_samples.items():
    # Construct the full image file path
    image_file_path = os.path.join(image_directory, f"heat_image_{timestamp}.png")

    # Read the image
    img = cv2.imread(image_file_path)

    # Initialize video writer with the size of the first image
    if video_writer is None:
        frame_size = (img.shape[1], img.shape[0])
        video_codec = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
        video_writer = cv2.VideoWriter(output_video_file, video_codec, frame_rate, frame_size)

    # Write the image to the video
    video_writer.write(img)

# Release the video writer
video_writer.release()