import os
import argparse

# Numpy imports
import numpy as np

# PyTorch imports
import torch
from torchvision.ops import box_iou

# Default YOLOv8 imports
from ultralytics import YOLO

# Wandb imports
import wandb

# Dataset preparation and autolabeling imports
from prepare_dataset import PrepareDataset
from autolabel import AutoLabeling


class ModelControler():
    '''
    Train the YOLOv8 model on the Water/Land detection dataset.

    Functions:
    - __init__(self, opt): Initialize the model controler
    - prepare_dataset_folder(self): Prepare the dataset folder
    '''

    def __init__(self, opt):
        # Set the command line arguments
        self.opt = opt

        # Dataset initialization
        self.dataset_preparation_class = PrepareDataset(self.opt.dataset_root, self.opt.train_val_ratio, self.opt.train_test_ratio)

        # AutoLabeling initialization
        self.autolabel_class = AutoLabeling(self.opt.dataset_root)

        # Model hyperparameter and device initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        # Print pytorch version
        print(f"PyTorch version: {torch.__version__}")

    # Function to initialize a new run
    def initialize_new_run(self, name):
        wandb.init(project="U-Net-Water-Land-Segmentation", name=name)

    # Function to prepare the dataset folder
    def prepare_dataset_folder(self):
        # Split the dataset into 3 subdatasets
        self.dataset_preparation_class.split_dataset()
        # Threshold the contour images
        self.dataset_preparation_class.threshold_contour_images()
        # Resize the images in dataset
        if self.opt.resize_prepared:
            self.dataset_preparation_class.resize_images(self.opt.resize_prepared_size, self.opt.resize_prepared_preserve_aspect_ratio)

    def autolabel_dataset(self):
        # Intialize the annotation folders
        self.autolabel_class.initialize_label_folders()
        # Label the ground truth using rawcontours or autodistil
        if self.opt.autolabel_method == 'rawcontours':
            self.autolabel_class.label_raw_and_contours()
        elif self.opt.autolabel_method == 'autodistil':
            self.autolabel_class.label_autodistil()
        else:
            print("Invalid autolabeling method")
            exit(1)

    def train_val_contour(self):
        # Initialize a new run
        self.initialize_new_run("YOLOv8-contour-model-train")

        # Define the dataset configuration
        contour_ds = self.opt.dataset_root + '/contour-det/config.yaml'
        # Load the YOLOv8 model depending on the size
        if self.opt.model_size == 'n':
            model = YOLO('/app/container/detection/weights/yolov8n.pt')
        elif self.opt.model_size == 's':
            model = YOLO('/app/container/detection/weights/yolov8s.pt')
        elif self.opt.model_size == 'm':
            model = YOLO('/app/container/detection/weights/yolov8m.pt')
        elif self.opt.model_size == 'l':
            model = YOLO('/app/container/detection/weights/yolov8l.pt')
        elif self.opt.model_size == 'x':
            model = YOLO('/app/container/detection/weights/yolov8x.pt')
        else:
            print("Invalid model size")
            exit(1)

        # Train the model
        if self.device == 'cpu':
            model.train(data=contour_ds, imgsz=self.opt.resize_prepared_size[0], epochs=self.opt.epochs, batch=-1, device=self.device)
        else:
            model.train(data=contour_ds, imgsz=self.opt.resize_prepared_size[0], epochs=self.opt.epochs, batch=-1, device='0')
        # Stop wandb logging
        wandb.finish()

        # Initialize a new run
        self.initialize_new_run("YOLOv8-contour-model-val")

        # Validate the model
        if self.device == 'cpu':
            metrics = model.val(data=contour_ds, batch=-1, device=self.device)
        else:
            metrics = model.val(data=contour_ds, batch=-1, device='0')

        print(metrics)

        # Stop wandb logging 
        wandb.finish()


    def test_contour(self):
        # TODO
        pass

    def train_val_autodistil(self):
        #TODO
        pass

    def test_autodistil(self):
        #TODO
        pass

    def test_pretrained(self):
        # Initialize a new run
        self.initialize_new_run("YOLOv8-pretrained-model-test")
        
        # Define the test data paths
        test_images = self.opt.dataset_root + '/raw-det/images'
        test_labels = self.opt.dataset_root + '/raw-det/labels'
        # Load the YOLOv8 model depending on the size
        if self.opt.model_size == 'n':
            model = YOLO('/app/container/detection/weights/yolov8n.pt')
        elif self.opt.model_size == 's':
            model = YOLO('/app/container/detection/weights/yolov8s.pt')
        elif self.opt.model_size == 'm':
            model = YOLO('/app/container/detection/weights/yolov8m.pt')
        elif self.opt.model_size == 'l':
            model = YOLO('/app/container/detection/weights/yolov8l.pt')
        elif self.opt.model_size == 'x':
            model = YOLO('/app/container/detection/weights/yolov8x.pt')
        else:
            print("Invalid model size")
            exit(1)
        # Validate the model
        if self.device == 'cpu':
            results = model.predict(source=test_images, stream=True, device=self.device)
        else:
            results = model.predict(source=test_images, stream=True, device='0')

        # Initialize lists to store IoU and Dice score for each image
        iou_list = []
        dice_score_list = []
        # Iterate over the results
        for result in results:
            # Get the ground truth path
            image_filename = os.path.splitext(os.path.basename(result.path))[0]
            # Set the label file path
            label_file_path = os.path.join(test_labels, f'{image_filename}.txt')
            # Check if the label file exists
            if os.path.exists(label_file_path):
                # Read ground truth bounding box coordinates from the label file
                with open(label_file_path, 'r') as label_file:
                    ground_truth = [list(map(float, line.split())) for line in label_file.readlines()]
                
                # Get the predicted bounding box coordinates and confidence scores
                pred_boxes = result.boxes.numpy()  # Convert to numpy array
                # Compare the prediction with the ground truth
                evaluation_results = ModelControler.compare_predictions(pred_boxes, ground_truth)

                # Append IoU and Dice score to the lists
                iou_list.append(evaluation_results['test/image_iou'])
                dice_score_list.append(evaluation_results['test/image_dice_score'])
                # Log the evaluation results
                wandb.log(evaluation_results)
            # If the label file does not exist set the IoU and Dice score to the worst possible value
            else: 
                iou_list.append(0.0)
                dice_score_list.append(0.0)

        # Calculate mean IoU and mean Dice score
        mean_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0.0
        mean_dice_score = sum(dice_score_list) / len(dice_score_list) if len(dice_score_list) > 0 else 0.0
        # Log mean IoU and mean Dice score
        wandb.log({'test/mean_iou': mean_iou, 'test/mean_dice_score': mean_dice_score})
        # Stop wandb logging
        wandb.finish()
        
    @staticmethod
    def calculate_iou(box1, box2):
        # Calculate IoU using torchvision's box_iou function
        boxes1 = torch.tensor([box1], dtype=torch.float32)
        boxes2 = torch.tensor([box2], dtype=torch.float32)
        iou = box_iou(boxes1, boxes2).item()
        return iou

    @staticmethod
    def calculate_dice_score(box1, box2):
        # Calculate Dice score
        intersection = ModelControler.calculate_intersection(box1, box2)
        union = ModelControler.calculate_union(box1, box2)
        dice_score = 2 * intersection / (intersection + union)
        return dice_score

    @staticmethod
    def calculate_intersection(box1, box2):
        # Calculate intersection area
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x_intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_intersection = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_intersection * y_intersection
        return intersection

    @staticmethod
    def calculate_union(box1, box2):
        # Calculate union area
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - ModelControler.calculate_intersection(box1, box2)
        return union

    @staticmethod
    def compare_predictions(pred_boxes, ground_truth):
        # Initialize lists to store IoU and Dice score for each prediction
        iou_list = []
        dice_score_list = []

        # Iterate over all ground truth boxes
        for gt_box in ground_truth:
            # Initialize lists to store IoU and Dice score for the current ground truth box
            iou_per_gt = []
            dice_score_per_gt = []
            
            # Check if there are no predicted boxes
            if len(pred_boxes) == 0:
                iou_list.append(0.0)
                dice_score_list.append(0.0)
            # If there are predicted boxes compare them with the ground truth
            else:
                # Iterate over all predicted boxes for images
                for pred_box in pred_boxes:
                    pred_box = pred_box.xyxy

                    # Iterate over sepparate predicted boxes if there are multiple
                    for item in pred_box:
                        pred_box = item
                        gt_box = np.array(gt_box)
                        # Calculate IoU and Dice score for the current boxes
                        iou = ModelControler.calculate_iou(pred_box, gt_box)
                        dice_score = ModelControler.calculate_dice_score(pred_box, gt_box)
                        # Append IoU and Dice score to the lists for the current predicted box
                        iou_per_gt.append(iou)
                        dice_score_per_gt.append(dice_score)

                # Choose the predicted box with the highest IoU for the current ground truth box
                max_iou_index = iou_per_gt.index(max(iou_per_gt))
                iou_list.append(iou_per_gt[max_iou_index])
                dice_score_list.append(dice_score_per_gt[max_iou_index])

        # Calculate mean IoU and mean Dice score for all ground truth boxes
        mean_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0.0
        mean_dice_score = sum(dice_score_list) / len(dice_score_list) if len(dice_score_list) > 0 else 0.0
        # Return the mean IoU and mean Dice score as images IoU and Dice score
        return {'test/image_iou': mean_iou, 'test/image_dice_score': mean_dice_score}


if __name__ == "__main__":
    # Arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Train a model for image detection')

    # Dataset control
    options.add_argument('--prepare', action='store_true', help='Prepare the dataset')
    options.add_argument('--resize-prepared', action='store_true', help='Resize the images and masks')
    options.add_argument('--resize-prepared-preserve-aspect-ratio', action='store_true', help='Resize the images and masks while preserving aspect ratio')
    options.add_argument('--resize-prepared-size', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    # Autolabeling control
    options.add_argument('--autolabel', action='store_true', help='Autolabel the dataset')
    options.add_argument('--autolabel-method', type=str, default='contours', help='Method to use for autolabeling can be either: rawcontours/autodistil')
    # Model control
    options.add_argument('--model-size', type=str, default='s', help='Size of the YOLOv8 model can be either: n/s/m/l/x')
    options.add_argument('--train', action='store_true', help='Train the model')
    options.add_argument('--train-method', type=str, default='contours', help='Method to use for training the YOLOv8 model can be either: contours/autodistil')
    options.add_argument('--test', action='store_true', help="Test the model")
    options.add_argument('--test-method', type=str, default='pretrained', help='Method to use for testing the YOLOv8 model can be either: pretrained/contours/autodistil')
    # Configuration
    options.add_argument('--dataset-root', type=str, default='/app/container/dataset', help='Path to the dataset root folder')
    options.add_argument('--train-test-ratio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    options.add_argument('--train-val-ratio', type=float, default=0.5, help='Ratio of the training dataset to be used for validation')
    options.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    opt = options.parse_args()

    # Model controler initialization
    model_controler = ModelControler(opt)

    # Prepare the dataset
    if opt.prepare:
        print("Preparing the detection dataset...")
        model_controler.prepare_dataset_folder()

    # Autolabel the dataset
    if opt.autolabel:
        print("Autolabeling the detection dataset...")
        model_controler.autolabel_dataset()

    # Train the model and generate the weights
    if opt.train_method == 'contours':
        print("Training the detection model using contours...")
        model_controler.train_val_contour()

    if opt.train_method == 'autodistil':
        print("Training the detection model using autodistil...")
        model_controler.train_val_autodistil()

    # Test the model and evaluate it
    if opt.test_method == 'pretrained':
        print("Testing the pretrained detection model...")
        model_controler.test_pretrained()

    if opt.test_method == 'contours':
        print("Testing the contour trained detection model...")
        model_controler.test_contour()

    if opt.test_method == 'autodistil':
        print("Testing the autodistil trained detection model...")
        model_controler.test_autodistil()