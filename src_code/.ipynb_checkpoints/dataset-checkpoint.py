import glob
import os
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import pandas as pd
import cv2
import numpy as np
import torch
from torchvision import transforms
class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])

        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        h,w=image.shape[0:2]
        
        image=image/255.0
        image=cv2.resize(image,(224,224))
        image=np.transpose(image,(2,0,1))
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        key_pts = key_pts.astype("float32").reshape(-1, 2)
        key_pts=key_pts*[224/w,224/h]
        sample = {"image": torch.tensor(image,dtype=torch.float), "keypoints":torch.tensor(key_pts,dtype=torch.float)}

        if self.transform:
            sample = self.transform(sample)

        return sample