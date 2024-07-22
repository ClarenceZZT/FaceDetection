import pandas as pd
import numpy as np
from PIL import Image
import os
import math
import shutil
from tqdm import tqdm

# Function to rotate a point around another point
def rotate_point(point, angle, center):
    angle_rad = math.radians(angle)
    ox, oy = center
    px, py = point

    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return qx, qy


file_path = '../data/training_frames_keypoints.csv'
keypoints_data = pd.read_csv(file_path)
image_dir = '../data/training/'
output_dir = '../data/training_aug/'


if os.path.exists(output_dir):
    shutil.rmtree(output_dir)


os.makedirs(output_dir, exist_ok=True)


rotation_probability = 0.5


for index, row in tqdm(keypoints_data.iterrows(), total=keypoints_data.shape[0], desc="Processing Images"):
    image_name = row['Unnamed: 0']
    image_path = os.path.join(image_dir, image_name)
    keypoints = row[1:].values.reshape(-1, 2)

    # Load image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')


    if np.random.rand() < rotation_probability:
        angle = np.random.uniform(0, 360)
        rotated_image = image.rotate(-angle)
        image_center = (image.width / 2, image.height / 2)
        rotated_keypoints = [rotate_point(point, angle, image_center) for point in keypoints]
        keypoints_data.loc[index, 1:] = np.array(rotated_keypoints).flatten()
    else:
        rotated_image = image

    rotated_image_path = os.path.join(output_dir, image_name)
    rotated_image.save(rotated_image_path)

# Save the updated keypoints dataset
output_csv_path = os.path.join('../data/', 'rotated_keypoints.csv')

if os.path.exists(output_csv_path):
    os.remove(output_csv_path)

keypoints_data.to_csv(output_csv_path, index=False)