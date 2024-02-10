import os
import shutil
import numpy as np
from PIL import Image


class PrepareDataset():
    '''
    Split the dataset into train, val and test directories from the initial root directory 
    with folders of images and masks called 'RGB' and 'WASR' respectively.

    Function arguments:
    root: str
        The root directory of the dataset
    train_test_ratio: float
        The ratio of the dataset to be used for training and testing
    train_val_ratio: float
        The ratio of the training dataset to be used for validation
    '''

    def __init__(self, root, train_val_ratio, train_test_ratio):
        # Initialize the root directory and the train and test ratios
        self.root = root
        self.train_val_ratio = train_val_ratio
        self.train_test_ratio = train_test_ratio
        # Initialize the train, val and test lists
        self.train = []
        self.val = []
        self.test = []
        # Initialize the threshold color
        self.threshold_error = 1
        self.threshold_color = np.array([41, 167, 224], dtype=np.uint8)

    def split_dataset(self):
        # Create train and test folders in the root directory
        os.makedirs(os.path.join(self.root, 'train-det'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'val-det'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'test-det'), exist_ok=True)

        # Create the images and masks folders in the train and test directories
        for folders in ['images', 'masks-grayscale', 'masks-rgb', 'gt-images', 'gt-masks-rgb', 'gt-masks-grayscale']:
            os.makedirs(os.path.join(self.root, 'train-det', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'val-det', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'test-det', folders), exist_ok=True)

        # Calculate the number of train and test images
        num_train_val = int(len(os.listdir(os.path.join(self.root, 'RGB'))) * self.train_test_ratio)
        num_train = int(num_train_val * self.train_val_ratio)
        num_val = num_train_val - num_train
        num_test = len(os.listdir(os.path.join(self.root, 'RGB'))) - num_train_val

        # Split the dataset
        for i, file in enumerate(os.listdir(os.path.join(self.root, 'RGB'))):
            if i < num_train:
                self.train.append(file)
            elif i < num_train_val:
                self.val.append(file)
            else:
                self.test.append(file)

        # Copy the images and masks to the train, val and test directories
        for file in self.train:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'train-det', 'images'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'train-det', 'gt-images'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'train-det', 'masks-grayscale'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'train-det', 'gt-masks-grayscale'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'train-det', 'masks-rgb'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'train-det', 'gt-masks-rgb'))

        for file in self.val:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'val-det', 'images'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'val-det', 'gt-images'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'val-det', 'masks-grayscale'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'val-det', 'gt-masks-grayscale'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'val-det', 'masks-rgb'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'val-det', 'gt-masks-rgb'))

        for file in self.test:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'test-det', 'images'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'test-det', 'gt-images'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'test-det', 'masks-grayscale'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'test-det', 'gt-masks-grayscale'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'test-det', 'masks-rgb'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'test-det', 'gt-masks-rgb'))

        # Print size of train, val and test datasets
        print(f"Train images: {len(self.train)}")
        print(f"Val images: {len(self.val)}")
        print(f"Test images: {len(self.test)}")

    def threshold_masks_grayscale(self):
        # Define the mask folders to threshold
        mask_folders = ['masks-grayscale', 'gt-masks-grayscale']
        # Threshold the masks folder using the grayscale threshold and convert to grayscale
        for folder in mask_folders:
            for file in os.listdir(os.path.join(self.root, 'train-det', folder)):
                mask_path = os.path.join(self.root, 'train-det', folder, file)
                mask = Image.open(mask_path).convert('RGB')
                mask_array = np.array(mask)
                mask_diff = np.abs(mask_array - self.threshold_color)
                mask = np.all(mask_diff < self.threshold_error, axis=-1)
                mask = np.where(mask, 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(mask, mode="L") 
                mask_image.save(mask_path)

            for file in os.listdir(os.path.join(self.root, 'val-det', folder)):
                mask_path = os.path.join(self.root, 'val-det', folder, file)
                mask = Image.open(mask_path).convert('RGB')
                mask_array = np.array(mask)
                mask_diff = np.abs(mask_array - self.threshold_color)
                mask = np.all(mask_diff < self.threshold_error, axis=-1)
                mask = np.where(mask, 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(mask, mode="L")
                mask_image.save(mask_path)

            for file in os.listdir(os.path.join(self.root, 'test-det', folder)):
                mask_path = os.path.join(self.root, 'test-det', folder, file)
                mask = Image.open(mask_path).convert('RGB')
                mask_array = np.array(mask)
                mask_diff = np.abs(mask_array - self.threshold_color)
                mask = np.all(mask_diff < self.threshold_error, axis=-1)
                mask = np.where(mask, 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(mask, mode="L")
                mask_image.save(mask_path)

    def threshold_masks_rgb(self):
        # Define the mask folders to threshold
        mask_folders = ['masks-rgb', 'gt-masks-rgb']
        # Threshold the masks folder using the color threshold and convert to grayscale
        for folder in mask_folders:
            for file in os.listdir(os.path.join(self.root, 'train-det', folder)):
                mask_path = os.path.join(self.root, 'train-det', folder, file)
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                mask = np.where(np.all(mask_array > self.threshold_color, axis=-1), 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(mask, mode="L")
                mask_image.save(mask_path)

            for file in os.listdir(os.path.join(self.root, 'val-det', folder)):
                mask_path = os.path.join(self.root, 'val-det', folder, file)
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                mask = np.where(np.all(mask_array > self.threshold_color, axis=-1), 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(mask, mode="L")
                mask_image.save(mask_path)

            for file in os.listdir(os.path.join(self.root, 'test-det', folder)):
                mask_path = os.path.join(self.root, 'test-det', folder, file)
                mask = Image.open(mask_path)
                mask_array = np.array(mask)
                mask = np.where(np.all(mask_array > self.threshold_color, axis=-1), 255, 0).astype(np.uint8)
                mask_image = Image.fromarray(mask, mode="L")
                mask_image.save(mask_path)

    def resize_images_and_masks(self, size, preserve_aspect_ratio):
        # Resize all images, masks and gt to the specified size
        folders_to_resize = ['images', 'masks-grayscale', 'masks-rgb', 'gt-images', 'gt-masks-rgb', 'gt-masks-grayscale']
        # Resize the images and masks
        for folder in folders_to_resize:
            if preserve_aspect_ratio:
                # Resize images in the images folder to the specified size while preserving aspect ratio
                for file in os.listdir(os.path.join(self.root, 'train-det', folder)):
                    image_path = os.path.join(self.root, 'train-det', folder, file)
                    image = Image.open(image_path)
                    image.thumbnail(size)
                    image.save(image_path)

                for file in os.listdir(os.path.join(self.root, 'val-det', folder)):
                    image_path = os.path.join(self.root, 'val-det', folder, file)
                    image = Image.open(image_path)
                    image.thumbnail(size)
                    image.save(image_path)

                for file in os.listdir(os.path.join(self.root, 'test-det', folder)):
                    image_path = os.path.join(self.root, 'test-det', folder, file)
                    image = Image.open(image_path)
                    image.thumbnail(size)
                    image.save(image_path)
            else:
                # Resize images in the images folder to the specified size
                for file in os.listdir(os.path.join(self.root, 'train-det', folder)):
                    image_path = os.path.join(self.root, 'train-det', folder, file)
                    image = Image.open(image_path).resize(size)
                    image.save(image_path)

                for file in os.listdir(os.path.join(self.root, 'val-det', folder)):
                    image_path = os.path.join(self.root, 'val-det', folder, file)
                    image = Image.open(image_path).resize(size)
                    image.save(image_path)

                for file in os.listdir(os.path.join(self.root, 'test-det', folder)):
                    image_path = os.path.join(self.root, 'test-det', folder, file)
                    image = Image.open(image_path).resize(size)
                    image.save(image_path)