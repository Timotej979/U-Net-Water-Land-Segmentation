import os
import shutil
import numpy as np
from PIL import Image

class PrepareDataset():
    '''
    Split the dataset into train and test directories from the initial root directory 
    with folders of images and masks called 'RGB' and 'WASR' respectively.

    Function arguments:
    root: str
        The root directory of the dataset
    train_ratio: float
        The ratio of the dataset to be used for training
    '''

    def __init__(self, root, train_ratio):
        # Initialize the root directory and the train and test ratios
        self.root = root
        self.train_ratio = train_ratio
        self.test_ratio = 1 - train_ratio
        # Initialize the train and test lists
        self.train = []
        self.test = []
        # Initialize the threshold color
        self.threshold_error = 1
        self.threshold_color = np.array([41, 167, 224], dtype=np.uint8)

    def split_dataset(self):
        # Create train and test folders in the root directory
        os.makedirs(os.path.join(self.root, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'test'), exist_ok=True)

        # Create the images and masks folders in the train and test directories
        for folders in ['images', 'masks']:
            os.makedirs(os.path.join(self.root, 'train', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'test', folders), exist_ok=True)

        # Calculate the number of train and test images
        num_train = int(len(os.listdir(os.path.join(self.root, 'RGB'))) * self.train_ratio)
        num_test = len(os.listdir(os.path.join(self.root, 'RGB'))) - num_train

        # Split the dataset
        for i, file in enumerate(os.listdir(os.path.join(self.root, 'RGB'))):
            if i < num_train:
                self.train.append(file)
            else:
                self.test.append(file)

        # Copy the images and masks to the train and test directories
        for file in self.train:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'train', 'images'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'train', 'masks'))

        for file in self.test:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'test', 'images'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'test', 'masks'))

    def threshold_masks(self):
        # Threshold images in the masks folder using the color and error threshold
        for file in os.listdir(os.path.join(self.root, 'train', 'masks')):
            mask_path = os.path.join(self.root, 'train', 'masks', file)
            mask = Image.open(mask_path).convert("RGB")
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask)
            mask_image.save(mask_path)

        for file in os.listdir(os.path.join(self.root, 'test', 'masks')):
            mask_path = os.path.join(self.root, 'test', 'masks', file)
            mask = Image.open(mask_path).convert("RGB")
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask)
            mask_image.save(mask_path)