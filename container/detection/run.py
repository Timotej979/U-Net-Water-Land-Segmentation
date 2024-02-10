import argparse



from prepare_dataset import PrepareDataset



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

    # Function to prepare the dataset folder
    def prepare_dataset_folder(self):
        # Split the dataset into train and test directories
        self.dataset_preparation_class.split_dataset()
        # Threshold the masks
        self.dataset_preparation_class.threshold_masks_grayscale()
        self.dataset_preparation_class.threshold_masks_rgb()
        # Resize the images and masks
        if self.opt.resize_prepaired:
            self.dataset_preparation_class.resize_images_and_masks(self.opt.resize_prepaired_size, self.opt.resize_prepaired_preserve_aspect_ratio)










if __name__ == "__main__":
    # Arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Train a model for image detection')

    # Dataset control
    options.add_argument('--prepare', action='store_true', help='Prepare the dataset')
    options.add_argument('--resize-prepaired', action='store_true', help='Resize the images and masks')
    options.add_argument('--resize-prepaired-preserve-aspect-ratio', action='store_true', help='Resize the images and masks while preserving aspect ratio')
    options.add_argument('--resize-prepaired-size', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    # Model control
    options.add_argument('--train', action='store_true', help='Train the model')
    options.add_argument('--test', action='store_true', help="Test the model")
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
