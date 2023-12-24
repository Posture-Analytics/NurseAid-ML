import cv2
import json
import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import utils

HISTOGRAM_RESOLUTION = 0.5 # 0.1°C
    
# Directory where the images are stored
HEAT_IMAGE_DIR = './heat_images'

decoded_samples = utils.load_samples('samples.json')

utils.generate_images(decoded_samples, HISTOGRAM_RESOLUTION)

utils.generate_video_from_images(HEAT_IMAGE_DIR, 'video.avi', 5)