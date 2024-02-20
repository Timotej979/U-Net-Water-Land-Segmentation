import argparse

from ultralytics import YOLO

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
        self.dataset_preparation_class = PrepareDataset(self.opt.datasetRoot, self.opt.trainValRatio, self.opt.trainTestRatio)

        # AutoLabeling initialization
        self.autolabel_class = AutoLabeling(self.opt.datasetRoot)

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
        # Label the ground truth using contour or autodistil detection
        if self.opt.autolabel_method == 'contours':
            self.autolabel_class.label_contours()
        elif self.opt.autolabel_method == 'autodistil':
            self.autolabel_class.label_autodistil()
        else:
            print("Invalid autolabeling method")
            exit(1)

    def train_contour(self):
        #TODO
        pass

    def test_contour(self):
        # TODO
        pass

    def train_autodistil(self):
        #TODO
        pass

    def test_autodistil(self):
        #TODO
        pass

    def test_pretrained(self):
        #TODO
        pass



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
    options.add_argument('--autolabel-method', type=str, default='contours', help='Method to use for autolabeling can be either: contours/autodistil')
    # Model control
    options.add_argument('--train', action='store_true', help='Train the model')
    options.add_argument('--train-method', type=str, default='contours', help='Method to use for training the YOLOv8 model can be either: contours/autodistil')
    options.add_argument('--test', action='store_true', help="Test the model")
    options.add_argument('--test-method', type=str, default='pretrained', help='Method to use for testing the YOLOv8 model can be either: pretrained/contours/autodistil')
    # Configuration
    options.add_argument('--datasetRoot', type=str, default='/app/container/dataset', help='Root directory of the dataset')
    options.add_argument('--trainTestRatio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    options.add_argument('--trainValRatio', type=float, default=0.5, help='Ratio of the training dataset to be used for validation')
    options.add_argument('--batchsize', type=int, default=2, help='Batch size')
    options.add_argument('--imagesize', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    options.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
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
        model_controler.train_contour()

    if opt.train_method == 'autodistil':
        model_controler.train_autodistil()

    # Test the model and evaluate it
    if opt.test_method == 'pretrained':
        model_controler.test_pretrained()

    if opt.test_method == 'contours':
        model_controler.test_contour()

    if opt.test_method == 'autodistil':
        model_controler.test_autodistil()