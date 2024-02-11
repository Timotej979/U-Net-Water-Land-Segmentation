import os
import cv2
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv


class AutoLabeling():
    '''
    AutoLabeling class is used to label the ground truth of the dataset using two approaches:
        - OpenCV contour detection on the masks with limiting the maximum contour area and then transfering the labels to the images
        - Autodistil labeling using the pre-trained bigger Grounded SAM model to label the images
    '''

    def __init__(self, root):
        # Initialize the root of the dataset
        self.root = root

        # Initialize the paths of the train test and validation sets
        self.train_path = os.path.join(self.root, 'train-det')
        self.val_path = os.path.join(self.root, 'val-det')
        self.test_path = os.path.join(self.root, 'test-det')

        # Set max contour area normalized to the mask size
        self.max_contour_area = 0.25
        
    # Initialize the annotation folders
    def initialize_annotation_folders(self):
        # Create the annotations folders
        os.makedirs(os.path.join(self.train_path, 'yolov8-contour-annotations'), exist_ok=True)
        os.makedirs(os.path.join(self.val_path, 'yolov8-contour-annotations'), exist_ok=True)
        os.makedirs(os.path.join(self.test_path, 'yolov8-contour-annotations'), exist_ok=True)

    # Label the ground truth using contour detection
    def label_gt_contours(self):
        # Label all images in the train set
        for img in os.listdir(os.path.join(self.train_path, 'masks')):
            # Read the mask, original image and gt-mask
            mask = cv2.imread(os.path.join(self.train_path, 'masks', img), cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(os.path.join(self.train_path, 'images', img.replace('.png', '.jpg')))
            gt_mask = cv2.imread(os.path.join(self.train_path, 'gt-masks-contour', img))
            # Get the mask size to limit the contours
            mask_size = mask.shape[0] * mask.shape[1]
            # Invert the mask
            mask = cv2.bitwise_not(mask)
            # Find contours in binary image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create a rectangle around the contours
            for contour in contours:
                # Get the area of the contour
                area = abs(cv2.contourArea(contour))
                if area < mask_size * self.max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the rectangle on the original image and the gt-mask
                    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(gt_mask, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    # Write the labels to the yolov8-annotations folder
                    with open(os.path.join(self.train_path, 'yolov8-contour-annotations', img.replace('.png', '.txt')), 'w') as file:
                        file.write(f'0 {x+w/2} {y+h/2} {w} {h}\n')  

            # Save the original image and gt-mask
            cv2.imwrite(os.path.join(self.train_path, 'images', img.replace('.png', '.jpg')), original)
            cv2.imwrite(os.path.join(self.train_path, 'gt-masks-contour', img), gt_mask)

              

        # Label all images in the validation set
        for img in os.listdir(os.path.join(self.val_path, 'masks')):  
            # Read the mask, original image and gt-mask
            mask = cv2.imread(os.path.join(self.val_path, 'masks', img), cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(os.path.join(self.val_path, 'images', img.replace('.png', '.jpg')))
            gt_mask = cv2.imread(os.path.join(self.val_path, 'gt-masks-contour', img))
            # Get the mask size to limit the contours
            mask_size = mask.shape[0] * mask.shape[1]
            # Invert the mask
            mask = cv2.bitwise_not(mask)
            # Find contours in binary image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create a rectangle around the contours
            for contour in contours:
                # Get the area of the contour
                area = abs(cv2.contourArea(contour))
                if area < mask_size * self.max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the rectangle on the original image and the gt-mask
                    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(gt_mask, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    # Write the labels to the yolov8-annotations folder
                    with open(os.path.join(self.val_path, 'yolov8-contour-annotations', img.replace('.png', '.txt')), 'w') as file:
                        file.write(f'0 {x+w/2} {y+h/2} {w} {h}\n')

            # Save the original image and gt-mask
            cv2.imwrite(os.path.join(self.val_path, 'images', img.replace('.png', '.jpg')), original)
            cv2.imwrite(os.path.join(self.val_path, 'gt-masks-contour', img), gt_mask)  

        # Label all images in the test set
        for img in os.listdir(os.path.join(self.test_path, 'masks')):
            # Read the mask, original image and gt-mask
            mask = cv2.imread(os.path.join(self.test_path, 'masks', img), cv2.IMREAD_GRAYSCALE)
            original = cv2.imread(os.path.join(self.test_path, 'images', img.replace('.png', '.jpg')))
            gt_mask = cv2.imread(os.path.join(self.test_path, 'gt-masks-contour', img))
            # Get the mask size to limit the contours
            mask_size = mask.shape[0] * mask.shape[1]
            # Invert the mask
            mask = cv2.bitwise_not(mask)
            # Find contours in binary image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create a rectangle around the contours
            for contour in contours:
                # Get the area of the contour
                area = abs(cv2.contourArea(contour))
                if area < mask_size * self.max_contour_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Draw the rectangle on the original image and the gt-mask
                    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(gt_mask, (x, y), (x+w, y+h), (0, 0, 255), 2)

                    # Write the labels to the yolov8-annotations folder
                    with open(os.path.join(self.test_path, 'yolov8-contour-annotations', img.replace('.png', '.txt')), 'w') as file:
                        file.write(f'0 {x+w/2} {y+h/2} {w} {h}\n')

            # Save the original image and gt-mask
            cv2.imwrite(os.path.join(self.test_path, 'images', img.replace('.png', '.jpg')), original)
            cv2.imwrite(os.path.join(self.test_path, 'gt-masks-contour', img), gt_mask)

    def label_gt_autodistil(self):
        # Initialize the caption ontology and assign a single class to all the objects
        ontology=CaptionOntology({
            "boat": "obstacle",
            "canoe": "obstacle",
            "kayak": "obstacle",
            "paddle": "obstacle",
            "sailboat": "obstacle",
            "ship": "obstacle",
            "yacht": "obstacle",
        })

        # Initialize the base model
        base_model = GroundedSAM(ontology=ontology)

        # Label the images in the train, validation and test sets
        folders_to_annote = [self.train_path, self.val_path, self.test_path]
        for folder in folders_to_annote:
            # Create the autodistil-annotations folders
            os.makedirs(os.path.join(folder, 'autodistil-annotations'), exist_ok=True)
            
            # Label the images using autodistil
            dataset = base_model.label(
                input_folder=os.path.join(folder, 'gt-images-autodistil'),
                extension=".jpg",
                output_folder=os.path.join(folder, 'gt-images-autodistil'),
            )




