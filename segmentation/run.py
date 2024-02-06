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
        self.dataset_preparation_class = PrepareDataset(self.opt.datasetRoot, self.opt.trainRatio)

        # Model initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Wandb initialization
        wandb.init(project="UNet_water_land_segmentation", entity="train & test")

    # Function to prepare the dataset
    def prepare_dataset(self):
        # Split the dataset into train and test directories
        self.dataset_preparation_class.split_dataset()
        # Threshold the masks
        self.dataset_preparation_class.threshold_masks()

    # Function to train the model
    def train(self):
        # Define the mean and standard deviation of the dataset

        PRE__MEAN = [0.5, 0.5, 0.5]
        PRE__STD = [0.5, 0.5, 0.5]
        
        # Define transforms for dataset augmentation
        image_and_mask_transform_train=A.Compose([A.Resize(self.opt.imagesize[0], self.opt.imagesize[1]),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                    ToTensorV2()])
        
        image_only_transform_train=A.Compose([A.Normalize(PRE__MEAN, PRE__STD),
                                            A.RandomBrightnessContrast()])
        
        image_and_mask_transform_test=A.Compose([A.Resize(self.opt.imagesize[0], self.opt.imagesize[1]),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                    ToTensorV2()])
        
        image_only_transform_test=A.Compose([A.Normalize(PRE__MEAN, PRE__STD)])
        
        # Define DatasetFolder and DataLoader
        train_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'train'),
                                   image_only_transform=image_only_transform_train,
                                   transform=image_and_mask_transform_train)

        test_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'test'),
                                  image_only_transform=image_only_transform_test,
                                  transform=image_and_mask_transform_test)

        trainloader = DataLoader(train_data, self.opt.batchsize, shuffle=True)
        testloader = DataLoader(test_data, self.opt.batchsize, shuffle=False)

        # Log the dataset statistics
        wandb.log({"Starting training..."})
        wandb.log({"Train dataset size": len(train_data), "Test dataset size": len(test_data)})
        wandb.log({"Batch size": self.opt.batchsize})
        wandb.log({"Image size": self.opt.imagesize})
        wandb.log({"Number of epochs": self.opt.epochs})
        wandb.log({"Device": self.device})
        wandb.log({"Dataset root": self.opt.datasetRoot})
        wandb.log({"Train ratio": self.opt.trainRatio})

        print(f"Train dataset stats: number of images: {len(train_data)}")
        print(f"Test dataset stats: number of images: {len(test_data)}")

        if not os.path.exists('/output'):
            os.mkdir('/output')
        logfile = os.path.join('/output/log.txt')

        # Load the CNN model
        model = UNet().to(self.device)

        # Initialize the loss function and iou metric
        l_bce = torch.nn.BCEWithLogitsLoss()
        jaccard = BinaryJaccardIndex(threshold=0.5).to(self.device)

        # Init softmax layer
        sigmoid = torch.nn.Sigmoid()

        # Conversion from BGR to single channel grayscale images - for masks
        to_gray = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        gray_kernel = torch.FloatTensor([[[[0.114]], [[0.587]], [[0.299]]]])
        to_gray.weight = torch.nn.Parameter(gray_kernel, requires_grad=False)

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

        # Initialize the lists to store the loss and accuracy over time
        train_loss_over_time = []
        val_loss_over_time = []
        val_iou_over_time = []

        # Initialize the best accuracy
        best_iou=0

        # Start training
        start_time = time.time()
        break_flag = False

        for epoch in range(self.opt.epochs):
            wandb.log({"Epoch": epoch+1})
            print("Training epoch: {}/{}".format(epoch+1, self.opt.epochs))

            model.train()

            avg_loss = 0

            try:
                # Train
                for images, masks, img_paths, mask_paths in tqdm(trainloader):

                    images, masks = images.to(self.device), masks.to(self.device)
                    optimizer.zero_grad() # Set all gradients to 0

                    predictions = model(images) # Feedforward
                    # out = softmax(predictions)
                    # out = sigmoid(predictions)
                    
                    loss=l_bce(predictions, masks) # Calculate the error of the current batch
                    avg_loss+=loss.cpu().detach().numpy()
                    loss.backward() # Calculate gradients with backpropagation
                    optimizer.step() # optimize weights for the next batch
                
                avg_loss = avg_loss/len(train_data)
                wandb.log({"Average train loss": avg_loss})
                print("Epoch {}: average  train loss: {:.7f}".format(epoch+1, avg_loss))
                train_loss_over_time.append(avg_loss)

                # Validation
                model.eval()
                iou = []
                avg_val_loss = []

                with torch.no_grad():
                    # for images, masks, img_paths, mask_paths in tqdm(testloader):
                    for images, masks, img_paths, mask_paths in testloader:
                        images, masks = images.to(self.device), masks.to(self.device)

                        predictions = model(images)
                        out = sigmoid(predictions)
                        
                        # Calculate IoU for training data
                        predicted_masks_bin = torch.sigmoid(out) > 0.5
                        iou.append(self.calculate_iou(predicted_masks_bin, masks).cpu().numpy())

                        avg_val_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                # Calculate the average IoU and loss over the validation set
                iou = np.mean(iou) 
                avg_val_loss = np.mean(avg_val_loss)
                val_iou_over_time.append(iou)
                val_loss_over_time.append(avg_val_loss)

                # save network weights when the accuracy is great than the best_acc
                if iou > best_iou:
                    wandb.log({"New best found"})
                    wandb.log({"Current best IoU": iou})
                    wandb.log({"Current best val loss": avg_val_loss})
                    wandb.log({"Current best epoch": epoch+1})
                    print(f"New best found. Current best IoU: {iou}")
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './output/CNN_weights.pth')
                    best_iou = iou

                wandb;log({"Average IOU": iou, "Best IOU": best_iou, "Val loss": avg_val_loss})
                print("Average IOU: {:.7f}     Best IOU: {:.7f}, val loss: {:.7f}".format(iou, best_iou, avg_val_loss))

                with open(logfile, 'a') as file:
                    # Convert data to a string and write it to the file
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
        plt.savefig(os.path.join('output', filename))
        
        # Plot equal error rate over time
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(val_iou_over_time[1:])), val_iou_over_time[1:], c="dodgerblue")
        plt.title("IoU per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        # plt.ylabel("EER, AUC", fontsize=18)
        plt.legend(['IoU'], fontsize=18)
        filename = f'iou.svg'
        plt.savefig( os.path.join('output', filename))

        plt.show()

    # Function to calculate the intersection over union of the model
    def calculate_iou(predictions, masks):
        # Calculate the intersection over union
        pred = torch.squeeze(predictions)
        gt = torch.squeeze(masks)
        # Convert to binary
        intersection = torch.logical_and(pred, gt)
        union = torch.logical_or(pred, gt)
        iou = torch.sum(intersection) / torch.sum(union)
        return iou.mean()

# MAIN
if __name__ == "__main__":

    # arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model control
    options.add_argument('--prepare', action='store_true', help='Prepare the dataset.')
    options.add_argument('--train', action='store_true', help='Train the model.')
    options.add_argument('--test', action='store_true', help="Test the model.")
    # Configuration
    options.add_argument('--datasetRoot', type=str, default='/dataset', help='Root directory of the dataset')
    options.add_argument('--trainRatio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    options.add_argument('--batchsize', type=int, default=2, help='Batch size')
    options.add_argument('--imagesize', type=int, default=(512,384), help='Size of the image (height, width)')
    options.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    opt = options.parse_args()

    # Initialize the model controler
    model_controler = ModelControler(opt)

    # Prepare the dataset
    if opt.prepare:
        print("Preparing the dataset...")
        model_controler.prepare_dataset()

    # Train the model
    if opt.train:
        print("Training the model...")
        model_controler.train()
    
    # Test the model
    if opt.test:
        print("Testing the model...")
        model_controler.test()

    