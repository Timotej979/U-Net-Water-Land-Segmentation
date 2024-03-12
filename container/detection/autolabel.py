import os
import cv2
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM


class AutoLabeling():
    '''
    AutoLabeling class is used to label the ground truth of the dataset using two approaches:
        - OpenCV contour detection on the masks with limiting the maximum contour area and then transfering the labels to the images
        - Autodistil labeling using the pre-trained bigger Grounded SAM model to label the images
    '''

    def __init__(self, root):
        # Initialize the root of the dataset
        self.root = root

        # Set max contour area normalized to the mask size
        self.max_contour_area = 0.25
        
    # Initialize the annotation folders
    def initialize_label_folders(self):
        # Create 2 custom YOLOv8 datasets for contour and autodistil
        os.makedirs(os.path.join(self.root, 'raw-det', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'contour-det', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.root, 'autodistil-det', 'labels'), exist_ok=True)

        # Create train val and test directories in the contour and auto-distil directories with the images and labels folders
        for folders in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.root, 'contour-det', 'labels', folders), exist_ok=True)
            os.makedirs(os.path.join(self.root, 'autodistil-det', 'labels', folders), exist_ok=True)

    # Label the ground truth using contour detection
    def label_raw_and_contours(self):
        # Label all images in the train set
        for img in os.listdir(os.path.join(self.root, 'contour-det', 'gt-rgb', 'train')):
            # Read the mask and original image
            mask = cv2.imread(os.path.join(self.root, 'contour-det', 'gt-gray', 'train', img), cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(os.path.join(self.root, 'contour-det', 'gt-rgb', 'train', img), cv2.IMREAD_COLOR)
            # Get the mask size to limit the contours
            mask_size = mask.shape[0] * mask.shape[1]
            # Invert the mask
            mask = cv2.bitwise_not(mask)
            # Find contours in binary image and check if there are no contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                # Create a rectangle around the contours
                for contour in contours:
                    # Get the area of the contour
                    area = abs(cv2.contourArea(contour))
                    if area < mask_size * self.max_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw the rectangle on the original image
                        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        # Write the normalized labels in YOLOv8 format to the labels folder for contour and just xyxy for raw datasets
                        with open(os.path.join(self.root, 'contour-det', 'labels', 'train', img.replace('.png', '.txt')), 'w') as file:
                            file.write(f'0 {(x + w / 2) / mask.shape[1]} {(y + h / 2) / mask.shape[0]} {w / mask.shape[1]} {h / mask.shape[0]}\n')
                        with open(os.path.join(self.root, 'raw-det', 'labels', img.replace('.png', '.txt')), 'w') as file:
                            file.write(f'{x} {y} {x+w} {y+h}\n')

                # Save the original image
                cv2.imwrite(os.path.join(self.root, 'contour-det', 'gt-rgb', 'train', img), original)

        # Label all images in the validation set
        for img in os.listdir(os.path.join(self.root, 'contour-det', 'gt-rgb', 'val')):
            # Read the mask and original image
            mask = cv2.imread(os.path.join(self.root, 'contour-det', 'gt-gray', 'val', img), cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(os.path.join(self.root, 'contour-det', 'gt-rgb', 'val', img), cv2.IMREAD_COLOR)
            # Get the mask size to limit the contours
            mask_size = mask.shape[0] * mask.shape[1]
            # Invert the mask
            mask = cv2.bitwise_not(mask)
            # Find contours in binary image and check if there are no contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                # Create a rectangle around the contours
                for contour in contours:
                    # Get the area of the contour
                    area = abs(cv2.contourArea(contour))
                    if area < mask_size * self.max_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw the rectangle on the original image
                        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        # Write the normalized labels in YOLOv8 format to the labels folder for contour and just xyxy for raw datasets
                        with open(os.path.join(self.root, 'contour-det', 'labels', 'val', img.replace('.png', '.txt')), 'w') as file:
                            file.write(f'0 {(x + w / 2) / mask.shape[1]} {(y + h / 2) / mask.shape[0]} {w / mask.shape[1]} {h / mask.shape[0]}\n')
                        with open(os.path.join(self.root, 'raw-det', 'labels', img.replace('.png', '.txt')), 'w') as file:
                            file.write(f'{x} {y} {x+w} {y+h}\n')

                # Save the original image
                cv2.imwrite(os.path.join(self.root, 'contour-det', 'gt-rgb', 'val', img), original)

        # Label all images in the test set
        for img in os.listdir(os.path.join(self.root, 'contour-det', 'gt-rgb', 'test')):
            # Read the mask and original image
            mask = cv2.imread(os.path.join(self.root, 'contour-det', 'gt-gray', 'test', img), cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(os.path.join(self.root, 'contour-det', 'gt-rgb', 'test', img), cv2.IMREAD_COLOR)
            # Get the mask size to limit the contours
            mask_size = mask.shape[0] * mask.shape[1]
            # Invert the mask
            mask = cv2.bitwise_not(mask)
            # Find contours in binary image and check if there are no contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                # Create a rectangle around the contours
                for contour in contours:
                    # Get the area of the contour
                    area = abs(cv2.contourArea(contour))
                    if area < mask_size * self.max_contour_area:
                        x, y, w, h = cv2.boundingRect(contour)
                        # Draw the rectangle on the original image
                        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        # Write the normalized labels in YOLOv8 format to the labels folder for contour and just xyxy for raw datasets
                        with open(os.path.join(self.root, 'contour-det', 'labels', 'test', img.replace('.png', '.txt')), 'w') as file:
                            file.write(f'0 {(x + w / 2) / mask.shape[1]} {(y + h / 2) / mask.shape[0]} {w / mask.shape[1]} {h / mask.shape[0]}\n')
                        with open(os.path.join(self.root, 'raw-det', 'labels', img.replace('.png', '.txt')), 'w') as file:
                            file.write(f'{x} {y} {x+w} {y+h}\n')

                # Save the original image
                cv2.imwrite(os.path.join(self.root, 'contour-det', 'gt-rgb', 'test', img), original)

        # Create the yaml config file for the contour dataset for YOLOv8
        with open(os.path.join(self.root, 'contour-det', 'config.yaml'), 'w') as file:
            file.write('train: /app/container/dataset/contour-det/images/train\n')
            file.write('val: /app/container/dataset/contour-det/images/val\n')
            file.write('test: /app/container/dataset/contour-det/images/test\n')
            file.write('nc: 1\n')
            file.write('names: [\'obstacle\']\n')

        print("Rawcontour detection labeling done")

    # Label the ground truth using autodistil
    def label_autodistil(self):
        # Initialize the base model
        base_model = GroundedSAM(onthology = CaptionOntology({"boat": "obstacle",
                                                            "canoe": "obstacle",
                                                            "kayak": "obstacle",
                                                            "paddle": "obstacle",
                                                            "sailboat": "obstacle",
                                                            "ship": "obstacle",
                                                            "yacht": "obstacle",
                                                            }))

        # Make the labeled directory
        os.makedirs(os.path.join(self.root, "autodistil-det-labeled"), exist_ok=True)

        # Label the images in the train, validation and test sets
        dataset = base_model.label(
            input_folder=os.path.join(self.root, "autodistil-det", "images"),
            extension=".jpg",
            output_folder=os.path.join(self.root, "autodistil-det-labeled")
        )

        print("Autodistil labeling done")



