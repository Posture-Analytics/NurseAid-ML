import cv2
import json
import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import base64_thermal_decoder as decoder

def format_time(timestamp):

    # Convert the string of a millisecond timestamp into a int of a second timestamp
    timestamp = int(timestamp) / 1000

    # Convert the timestamp into a datetime object
    timestamp = datetime.fromtimestamp(timestamp)

    # Convert the datetime object into a formatted string
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # Chop off the last 3 digits of the string (the microseconds) and add the timezone
    return time_str[:-3] + " GMT-3"

def load_samples(samples_file_path):
    # Read the json file with the samples into a dictionary
    with open(samples_file_path, 'r') as f:
        samples = json.load(f)

    decoded_samples = {}

    # Iterate over each sample
    for timestamp, encoded_string in samples.items():
        # Decode the string
        decoded_values = decoder.decode_base64(encoded_string, 0, 100.0)

        # Assign the decoded values to the dictionary
        decoded_samples[timestamp] = decoded_values

    return decoded_samples

def generate_images(decoded_samples, histogram_resolution=0.5, output_directory="./heat_images"):
        
    counter = 0
    total = len(decoded_samples)

    # Ensure the directory for saving images exists
    os.makedirs("./heat_images", exist_ok=True)

    # Iterate over each sample
    for timestamp, sample_values in decoded_samples.items():
        counter += 1
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
        bin_edges = np.arange(math.floor(min_temp), math.ceil(max_temp) + histogram_resolution, histogram_resolution)

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
        plt.savefig(f"{output_directory}/heat_image_{timestamp}.png")
        plt.close(fig)

def generate_video_from_images(image_directory, output_video_file, fps=1):
    print("Creating video from images...")

    # Initialize video writer
    video_writer = None

    # Iterate over each file on the directory
    for file_name in os.listdir(image_directory):
        # Construct the full image file path
        image_file_path = os.path.join(image_directory, file_name)

        # Read the image
        img = cv2.imread(image_file_path)

        # Initialize video writer with the size of the first image
        if video_writer is None:
            frame_size = (img.shape[1], img.shape[0])
            video_codec = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
            video_writer = cv2.VideoWriter(output_video_file, video_codec, fps, frame_size)

        # Write the image to the video
        video_writer.write(img)

    # Release the video writer
    video_writer.release()

# def get_pages_from_frame(frame_array):
    
#     # Creating the chessboard pattern arrays
#     page_0 = np.zeros((12, 32))
#     page_1 = np.zeros((12, 32))

#     # Iterate over the rows and collumns of the frame, filling the pages
#     for row in range(24):
#         for col in range(32):
#             if (row + col) % 2 == 0:
#                 page_0[row // 2][col] = frame_array[row][col]
#             else:
#                 page_1[row // 2][col] = frame_array[row][col]

#     return page_0, page_1

def get_pages_from_frame(frame_array, compression_direction='horizontal'):

    if compression_direction == 'vertical':
        # Creating the chessboard pattern arrays
        page_0 = np.zeros((12, 32))
        page_1 = np.zeros((12, 32))

        # Iterate over the rows and collumns of the frame, filling the pages
        for row in range(24):
            for col in range(32):
                if (row + col) % 2 == 0:
                    page_0[row // 2][col] = frame_array[row][col]
                else:
                    page_1[row // 2][col] = frame_array[row][col]

    elif compression_direction == 'horizontal':
        # Creating the chessboard pattern arrays
        page_0 = np.zeros((24, 16))
        page_1 = np.zeros((24, 16))

        # Iterate over the rows and collumns of the frame, filling the pages
        for row in range(24):
            for col in range(32):
                if (row + col) % 2 == 0:
                    page_0[row][col // 2] = frame_array[row][col]
                else:
                    page_1[row][col // 2] = frame_array[row][col]

    else:
        raise ValueError("Invalid compression direction. Valid values are 'horizontal' and 'vertical'.")
    
    return page_0, page_1

def get_frame_from_pages(page_0_array, page_1_array, compression_direction='horizontal'):
        
        # Creating the chessboard pattern arrays
        frame_array = np.zeros((24, 32))
    
        if compression_direction == 'vertical':
            # Iterate over the rows and collumns of the frame, filling the pages
            for row in range(24):
                for col in range(32):
                    if (row + col) % 2 == 0:
                        frame_array[row][col] = page_0_array[row // 2][col]
                    else:
                        frame_array[row][col] = page_1_array[row // 2][col]
    
        elif compression_direction == 'horizontal':
            # Iterate over the rows and collumns of the frame, filling the pages
            for row in range(24):
                for col in range(32):
                    if (row + col) % 2 == 0:
                        frame_array[row][col] = page_0_array[row][col // 2]
                    else:
                        frame_array[row][col] = page_1_array[row][col // 2]
    
        else:
            raise ValueError("Invalid compression direction. Valid values are 'horizontal' and 'vertical'.")
        
        return frame_array

def interpolate_black_pixels(pseudoframe):
    
    # Iterate over the rows and collumns of the pseudoframe
    for row in range(24):
        for col in range(32):
            if pseudoframe[row][col] == 0:

                surrounding_pixels = []

                # Get the surrounding pixels, if they exist
                if row > 0:
                    surrounding_pixels.append(pseudoframe[row - 1][col])

                if row < 23:
                    surrounding_pixels.append(pseudoframe[row + 1][col])

                if col > 0:
                    surrounding_pixels.append(pseudoframe[row][col - 1])

                if col < 31:
                    surrounding_pixels.append(pseudoframe[row][col + 1])

                # If there are no surrounding pixels, skip
                if len(surrounding_pixels) == 0:
                    continue
                else:
                    # Calculate the mean of the surrounding pixels
                    mean = np.mean(surrounding_pixels)

                    # Assign the mean to the current pixel
                    pseudoframe[row][col] = mean

    return pseudoframe