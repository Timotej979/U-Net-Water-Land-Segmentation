import os, time, datetime, argparse

import numpy as np
from tqdm import tqdm

import wandb

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryJaccardIndex

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from prepare_dataset import PrepareDataset
from custom_dataset import DatasetFolder
from network import UNet


class ModelControler():
    '''
    Train the UNet model on the Water/Land segmentation dataset.
    '''

    def __init__(self, opt):
        # Set the command line arguments
        self.opt = opt
        
        # Dataset initialization
        self.dataset_preparation_class = PrepareDataset(self.opt.datasetRoot, self.opt.trainValRatio, self.opt.trainTestRatio)

        # Model hyperparameter and device initialization
        self.learning_rate = 0.0001
        self.betas = (0.9, 0.999)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Wandb initialization
        wandb.init(project="U-Net-Water-Land-segmentation", entity="Segmentation", config={ # Model configuration
                                                                                            "hidden_layer_sizes": [64, 128, 256, 512, 1024],
                                                                                            "kernel_sizes": [3, 3, 3, 3, 3],
                                                                                            "activation": "ReLU",
                                                                                            "pool_sizes": [2, 2, 2, 2, 2],
                                                                                            "dropout": 0.5,
                                                                                            # Binary classification: water or land
                                                                                            "num_classes": 1,
                                                                                            # Training configuration
                                                                                            "learning_rate": self.learning_rate,
                                                                                            "betas": self.betas,
                                                                                            "epochs": self.opt.epochs,
                                                                                            "batch_size": self.opt.batchsize,
                                                                                        })

        # Load the dataset
        self.load_dataset()

    # Function to calculate the intersection over union of the model
    @staticmethod
    def calculate_iou(predictions, masks):
        # Calculate the intersection over union
        pred = torch.squeeze(predictions)
        gt = torch.squeeze(masks)
        # Convert to binary
        intersection = torch.logical_and(pred, gt)
        union = torch.logical_or(pred, gt)
        iou = torch.sum(intersection) / torch.sum(union)
        return iou.mean()

    #######################################################################################################################
    # Function to prepare the dataset folder
    def prepare_dataset_folder(self):
        # Split the dataset into train and test directories
        self.dataset_preparation_class.split_dataset()
        # Threshold the masks
        self.dataset_preparation_class.threshold_masks()
        # Resize the images and masks
        if self.opt.resize_prepaired:
            self.dataset_preparation_class.resize_images_and_masks(self.opt.resize_prepaired_size, self.opt.resize_prepaired_preserve_aspect_ratio)

    # Function to load the dataset
    def load_dataset(self):
        # Define the mean and standard deviation of the dataset
        PRE_MEAN = [0.5, 0.5, 0.5]
        PRE_STD = [0.5, 0.5, 0.5]
        
        # Define transforms for dataset augmentation
        image_and_mask_transform_train=A.Compose([A.Resize(self.opt.imagesize[0], self.opt.imagesize[1]),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                    ToTensorV2()])
        
        image_only_transform_train=A.Compose([A.Normalize(PRE_MEAN, PRE_STD),
                                            A.RandomBrightnessContrast()])
        
        image_and_mask_transform_test=A.Compose([A.Resize(self.opt.imagesize[0], self.opt.imagesize[1]),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                    ToTensorV2()])
        
        image_only_transform_test=A.Compose([A.Normalize(PRE_MEAN, PRE_STD)])
        
        # Initialize train, validation and test datasets
        train_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'train'),
                                   image_only_transform=image_only_transform_train,
                                   transform=image_and_mask_transform_train)

        val_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'val'),
                                 image_only_transform=image_only_transform_test,
                                 transform=image_and_mask_transform_test)

        test_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'test'),
                                 image_only_transform=image_only_transform_test,
                                 transform=image_and_mask_transform_test)

        self.trainloader = DataLoader(train_data, self.opt.batchsize, shuffle=True)
        self.valloader = DataLoader(val_data, self.opt.batchsize, shuffle=False)
        self.testloader = DataLoader(test_data, self.opt.batchsize, shuffle=False)

        print(f"Train dataset stats: number of images: {len(self.train_data)}")
        print(f"Validation dataset stats: number of images: {len(self.val_data)}")
        print(f"Test dataset stats: number of images: {len(self.test_data)}")

    # Function to train the model
    def train(self):
        # Create the output directory
        if not os.path.exists('./output'):
            os.mkdir('./output')
        logfile = os.path.join('./output/log.txt')

        # Load the CNN model, loss_function and optimizer
        model = UNet().to(self.device)
        l_bce = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=self.betas)

        # Initialize the lists to store the loss and accuracy over time
        train_loss_over_time = []
        val_loss_over_time = []
        val_iou_over_time = []

        # Initialize the best accuracy
        best_iou=0

        # Start training
        start_time = time.time()

        for epoch in range(self.opt.epochs):
            wandb.log({"Epoch": epoch+1})
            print("Training epoch: {}/{}".format(epoch+1, self.opt.epochs))

            model.train()

            avg_loss = 0

            try:
                # Train
                for images, masks, img_paths, mask_paths in tqdm(self.trainloader):

                    images, masks = images.to(self.device), masks.to(self.device)
                    optimizer.zero_grad() # Set all gradients to 0

                    predictions = model(images) # Feedforward
                    
                    loss=l_bce(predictions, masks) # Calculate the error of the current batch
                    avg_loss+=loss.cpu().detach().numpy()
                    loss.backward() # Calculate gradients with backpropagation
                    optimizer.step() # optimize weights for the next batch

                    # Log the results
                    wandb.log({"Train Loss": loss})
                
                avg_loss = avg_loss/len(train_data)
                print("Epoch {}: average  train loss: {:.7f}".format(epoch+1, avg_loss))
                train_loss_over_time.append(avg_loss)

                # Validation
                model.eval()
                iou = []
                avg_val_loss = []

                with torch.no_grad():
                    for images, masks, img_paths, mask_paths in self.valloader:
                        images, masks = images.to(self.device), masks.to(self.device)

                        # Feedforward, softmax
                        predictions = model(images)
                        predicted_masks_bin = torch.sigmoid(predictions) > 0.5

                        # Calculate IoU for validation data
                        iou.append(self.calculate_iou(predicted_masks_bin, masks).cpu().numpy())

                        avg_val_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                        # Log the results
                        wandb.log({"Validation Loss": avg_val_loss[-1]})
                        wandb.log({"Validation IoU": iou[-1]})

                # Calculate the average IoU and loss over the validation set
                iou = np.mean(iou) 
                avg_val_loss = np.mean(avg_val_loss)
                val_iou_over_time.append(iou)
                val_loss_over_time.append(avg_val_loss)

                # Save network weights when the accuracy is great than the best_acc
                if iou > best_iou:
                    print(f"New best found. Current best IoU: {iou}")
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './output/CNN_weights.pth')
                    best_iou = iou

                print("Average IOU: {:.7f}     Best IOU: {:.7f}, val loss: {:.7f}".format(iou, best_iou, avg_val_loss))

                # Log the results
                with open(logfile, 'a') as file:
                    row = f'Epoch {epoch}: Train loss: {avg_loss}, val loss: {avg_val_loss}, val iou: {iou}' '\n'
                    file.write(row)

            except KeyboardInterrupt:
                break

        print('\n----------------------------------------------------------------')
        # Print total training time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total training time: {elapsed_time}")

        # Plot loss over time
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(train_loss_over_time[1:])), train_loss_over_time[1:], c="dodgerblue")
        plt.plot(range(len(val_loss_over_time[1:])), val_loss_over_time[1:], c="r")
        plt.title("Loss per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        plt.ylabel("loss", fontsize=18)
        plt.legend(['Training loss', 'Validation loss'], fontsize=18)
        filename = f'loss.svg'
        plt.savefig(os.path.join('./output', filename))
        
        # Plot equal error rate over time
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(val_iou_over_time[1:])), val_iou_over_time[1:], c="dodgerblue")
        plt.title("IoU per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        # plt.ylabel("EER, AUC", fontsize=18)
        plt.legend(['IoU'], fontsize=18)
        filename = f'iou.svg'
        plt.savefig( os.path.join('./output', filename))

    # Function to test the model
    def test(self):
        # Load the CNN model, loss_function and weights
        model = UNet().to(self.device)
        l_bce = nn.BCEWithLogitsLoss()
        model.load_state_dict(torch.load('./output/CNN_weights.pth')['state_dict'])
        model.eval()

        # Initialize the lists to store the loss and accuracy over time
        iou = []
        avg_test_loss = []

        # Start testing
        start_time = time.time()

        with torch.no_grad():
            for images, masks, _, _ in tqdm(self.testloader):
                images, masks = images.to(self.device), masks.to(self.device)

                # Feedforward, softmax
                predictions = model(images)

                predicted_masks_bin = torch.sigmoid(images) > 0.5
                iou.append(self.calculate_iou(predicted_masks_bin, masks).cpu().numpy())

                avg_test_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                # Log the results
                wandb.log({"Test Loss": avg_test_loss[-1]})
                wandb.log({"Test IoU": iou[-1]})

        # Print total testing time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total testing time: {elapsed_time}")

        # Calculate the average IoU and loss over the test set
        iou = np.mean(iou)
        avg_test_loss = np.mean(avg_test_loss)

        print("Average Test IOU: {:.7f}, Test loss: {:.7f}".format(iou, avg_test_loss))





# MAIN
if __name__ == "__main__":
    # Arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset control
    options.add_argument('--prepare', action='store_true', help='Prepare the dataset.')
    options.add_argument('--resize-prepaired', action='store_true', help='Resize the images and masks.')
    options.add_argument('--resize-prepaired-preserve-aspect-ratio', action='store_true', help='Resize the images and masks while preserving aspect ratio.')
    options.add_argument('--resize-prepaired-size', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    # Model control
    options.add_argument('--train', action='store_true', help='Train the model.')
    options.add_argument('--test', action='store_true', help="Test the model.")
    # Configuration
    options.add_argument('--datasetRoot', type=str, default='/app/container/dataset', help='Root directory of the dataset')
    options.add_argument('--trainTestRatio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    options.add_argument('--trainValRatio', type=float, default=0.5, help='Ratio of the training dataset to be used for validation')
    options.add_argument('--batchsize', type=int, default=2, help='Batch size')
    options.add_argument('--imagesize', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    options.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    opt = options.parse_args()

    # Initialize the model controler
    model_controler = ModelControler(opt)

    # Prepare the dataset
    if opt.prepare:
        print("Preparing the dataset...")
        model_controler.prepare_dataset_folder()

    # Train the model
    if opt.train:
        print("Training the model...")
        model_controler.train()
    
    # Test the model
    if opt.test:
        print("Testing the model...")
        model_controler.test()

    