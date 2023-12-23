import cv2
import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import base64_thermal_decoder as decoder

# Read the json file with the samples into a dictionary
with open('samples.json', 'r') as f:
    samples = json.load(f)

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

# Iterate over each sample
for timestamp, sample_values in decoded_samples.items():
    print(f"Processing timestamp: {timestamp}")

    # Converting the list of temperatures into a 32x24 array
    temperature_array = np.array(sample_values).reshape(24, 32)

    fig = plt.figure()

    # Plotting the heat image
    plt.imshow(temperature_array, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f"Heat Image for Timestamp: {timestamp}")

    # Saving the figure
    plt.savefig(f"./heat_images/heat_image_{timestamp}.png")
    plt.close(fig)




# Open all images and convert to a video (1 image per second) using cv2
    
# # Directory where the images are stored
# image_directory = './heat_images'

# # Output video file
# output_video_file = 'thermal_video.avi'

# # Video properties
# frame_rate = 1  # 1 frame per second

# # Initialize video writer
# video_writer = None

# # Iterate over each image
# for timestamp, _ in decoded_samples.items():
#     # Construct the full image file path
#     image_file_path = os.path.join(image_directory, f"heat_image_{timestamp}.png")

#     # Read the image
#     img = cv2.imread(image_file_path)

#     # Initialize video writer with the size of the first image
#     if video_writer is None:
#         frame_size = (img.shape[1], img.shape[0])
#         video_codec = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
#         video_writer = cv2.VideoWriter(output_video_file, video_codec, frame_rate, frame_size)

#     # Write the image to the video
#     video_writer.write(img)

# # Release the video writer
# video_writer.release()