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
        # Create 3 custom YOLOv8 datasets for raw, contour and autodistil detection
        os.makedirs(os.path.join(self.root, 'raw-det'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'contour-mask-det'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'contour-rgb-det'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'autodistil-nolabel-det'), exist_ok=True)

        # Create train val and test directories in the raw, contour and autodistil directories with the images and labels folders
        os.makedirs(os.path.join(self.root, 'raw-det', 'images'), exist_ok=True)
        for folders in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.root, 'contour-mask-det', 'gt-rgb', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'contour-mask-det', 'gt-gray', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'contour-mask-det', 'images', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'contour-rgb-det', 'images', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'autodistil-nolabel-det', 'images', folders), exist_ok=True)

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

        # Copy the images and masks to the train, val and test directories respectively
        for file in self.train:
            # Images for raw and autodistil are RGB, while images for contour are WASR
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'raw-det', 'images'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'autodistil-nolabel-det', 'images', 'train'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'train'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'gt-gray', 'train'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'images', 'train'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'contour-rgb-det', 'images', 'train'))

        for file in self.val:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'raw-det', 'images'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'autodistil-nolabel-det', 'images', 'val'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'val'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'gt-gray', 'val'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'images', 'val'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'contour-rgb-det', 'images', 'val'))

        for file in self.test:
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'raw-det', 'images'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'autodistil-nolabel-det', 'images', 'test'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'test'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'gt-gray', 'test'))
            shutil.copy(os.path.join(self.root, 'WASR', file.replace(".jpg", ".png")), os.path.join(self.root, 'contour-mask-det', 'images', 'test'))
            shutil.copy(os.path.join(self.root, 'RGB', file), os.path.join(self.root, 'contour-rgb-det', 'images', 'test'))

        # Print size of train, val and test datasets
        print(f"Train images: {len(self.train)}")
        print(f"Val images: {len(self.val)}")
        print(f"Test images: {len(self.test)}")

    def threshold_contour_images(self):
        # Threshold the contour dataset images using the color threshold and convert to rgb
        for file in os.listdir(os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'train')):
            # Threshold the images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'images', 'train', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="RGB")
            mask_image.save(mask_path)
            # Threshold the ground truth images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'train', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="RGB")
            mask_image.save(mask_path)
            # Threshold the ground truth images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'gt-gray', 'train', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="L") 
            mask_image.save(mask_path)

        for file in os.listdir(os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'val')):
            # Threshold the images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'images', 'val', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="RGB")
            mask_image.save(mask_path)
            # Threshold the ground truth images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'val', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="RGB")
            mask_image.save(mask_path)
            # Threshold the ground truth images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'gt-gray', 'val', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="L") 
            mask_image.save(mask_path)

        for file in os.listdir(os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'test')):
            # Threshold the images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'images', 'test', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="RGB")
            mask_image.save(mask_path)
            # Threshold the ground truth images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'gt-rgb', 'test', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask[:, :, np.newaxis], [255, 255, 255], [0, 0, 0]).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="RGB")
            mask_image.save(mask_path)
            # Threshold the ground truth images
            mask_path = os.path.join(self.root, 'contour-mask-det', 'gt-gray', 'test', file)
            mask = Image.open(mask_path).convert('RGB')
            mask_array = np.array(mask)
            mask_diff = np.abs(mask_array - self.threshold_color)
            mask = np.all(mask_diff < self.threshold_error, axis=-1)
            mask = np.where(mask, 255, 0).astype(np.uint8)
            mask_image = Image.fromarray(mask, mode="L")
            mask_image.save(mask_path)

    def resize_images(self, size, preserve_aspect_ratio):
        # Resize all images in datasets
        for dataset in ['raw-det', 'contour-rgb-det', 'contour-mask-det', 'autodistil-nolabel-det']:
            # Resize the images of raw dataset
            if dataset == 'raw-det':
                for file in os.listdir(os.path.join(self.root, dataset, 'images')):
                    img_path = os.path.join(self.root, dataset, 'images', file)
                    img = Image.open(img_path)
                    if preserve_aspect_ratio:
                        img.thumbnail(size)
                    else:
                        img = img.resize(size)
                    img.save(img_path)
            # Resize the images for contour dataset
            elif dataset == 'contour-mask-det':
                for folder in ['train', 'val', 'test']:
                    # Images
                    for file in os.listdir(os.path.join(self.root, dataset, 'images', folder)):
                        img_path = os.path.join(self.root, dataset, 'images', folder, file)
                        img = Image.open(img_path)
                        if preserve_aspect_ratio:
                            img.thumbnail(size)
                        else:
                            img = img.resize(size)
                        img.save(img_path)
                    # Ground truth
                    for file in os.listdir(os.path.join(self.root, dataset, 'gt-rgb', folder)):
                        img_path = os.path.join(self.root, dataset, 'gt-rgb', folder, file)
                        img = Image.open(img_path)
                        if preserve_aspect_ratio:
                            img.thumbnail(size)
                        else:
                            img = img.resize(size)
                        img.save(img_path)
                    # Ground truth
                    for file in os.listdir(os.path.join(self.root, dataset, 'gt-gray', folder)):
                        img_path = os.path.join(self.root, dataset, 'gt-gray', folder, file)
                        img = Image.open(img_path)
                        if preserve_aspect_ratio:
                            img.thumbnail(size)
                        else:
                            img = img.resize(size)
                        img.save(img_path)
            # Resize the images for autodistil dataset
            else:
                # Images
                for folder in ['train', 'val', 'test']:
                    for file in os.listdir(os.path.join(self.root, dataset, 'images', folder)):
                        img_path = os.path.join(self.root, dataset, 'images', folder, file)
                        img = Image.open(img_path)
                        if preserve_aspect_ratio:
                            img.thumbnail(size)
                        else:
                            img = img.resize(size)
                        img.save(img_path)