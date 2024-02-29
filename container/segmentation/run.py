import os, time, datetime, argparse

import numpy as np
from tqdm import tqdm

import wandb

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_curve, auc, classification_report

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from prepare_dataset import PrepareDataset
from custom_dataset import DatasetFolder
from network import UNet


class ModelControler():
    '''
    Train the UNet model on the Water/Land segmentation dataset.

    Functions:
    - __init__(self, opt): Initialize the model controler
    - calculate_iou(predictions, masks): Calculate the intersection over union of the model (Static method)
    - prepare_dataset_folder(self): Prepare the dataset folder
    - load_dataset(self): Load the dataset
    - train(self): Train the model
    - test(self): Test the model
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
        print(f"Device: {self.device}")

        # Define the transformations
        self.define_transformations()

    # Function to calculate the intersection over union of the model
    @staticmethod
    def calculate_iou_f1(predictions, masks):
        # Calculate the intersection over union (IoU)
        pred = torch.squeeze(predictions)
        gt = torch.squeeze(masks)
        
        # Convert to binary
        intersection = torch.logical_and(pred, gt)
        union = torch.logical_or(pred, gt)
        iou = torch.sum(intersection) / torch.sum(union)
        
        # Calculate Dice score (F1 score)
        dice_score = 2 * torch.sum(intersection) / (torch.sum(pred) + torch.sum(gt))

        # Calculate Pixel Accuracy
        correct_pixels = torch.sum(pred == gt)
        total_pixels = torch.numel(pred)
        pixel_accuracy = correct_pixels / total_pixels
        
        return iou.mean(), dice_score.mean(), pixel_accuracy

    # Function to initialize a new run
    def initialize_new_run(self, name):
        wandb.init(project="U-Net-Water-Land-Segmentation", name=name, config={ # Model configuration
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

    #######################################################################################################################
    # Function to prepare the dataset folder
    def prepare_dataset_folder(self):
        # Split the dataset into train and test directories
        self.dataset_preparation_class.split_dataset()
        # Threshold the masks
        self.dataset_preparation_class.threshold_masks()
        # Resize the images and masks
        if self.opt.resize_prepared:
            self.dataset_preparation_class.resize_images_and_masks(self.opt.resize_prepared_size, self.opt.resize_prepared_preserve_aspect_ratio)

    # Function to define image transformations
    def define_transformations(self):
        # Define the mean and standard deviation of the dataset
        PRE_MEAN = [0.5, 0.5, 0.5]
        PRE_STD = [0.5, 0.5, 0.5]
        
        # Define transforms for dataset augmentation
        self.image_and_mask_transform_train=A.Compose([A.Resize(self.opt.imagesize[0], self.opt.imagesize[1]),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                    ToTensorV2()])
        
        self.image_only_transform_train=A.Compose([A.Normalize(PRE_MEAN, PRE_STD),
                                            A.RandomBrightnessContrast()])
        
        self.image_and_mask_transform_test=A.Compose([A.Resize(self.opt.imagesize[0], self.opt.imagesize[1]),
                                                A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                    ToTensorV2()])
        
        self.image_only_transform_test=A.Compose([A.Normalize(PRE_MEAN, PRE_STD)])

    # Function to train the model
    def train(self):
        # Initialize train and validation datasets
        train_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'train-seg'),
                                   image_only_transform=self.image_only_transform_train,
                                   transform=self.image_and_mask_transform_train)

        val_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'val-seg'),
                                 image_only_transform=self.image_only_transform_test,
                                 transform=self.image_and_mask_transform_test)

        print(f"Train dataset stats: number of images: {len(train_data)}")
        print(f"Validation dataset stats: number of images: {len(val_data)}")

        trainloader = DataLoader(train_data, self.opt.batchsize, shuffle=True)
        valloader = DataLoader(val_data, self.opt.batchsize, shuffle=False)

        # Create the output directory
        if not os.path.exists('./segmentation/output'):
            os.mkdir('./segmentation/output')
        logfile = os.path.join('./segmentation/output/log.txt')

        # Load the CNN model, loss_function and optimizer
        model = UNet().to(self.device)
        l_bce = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, betas=self.betas)

        # Initialize the lists to store the loss and accuracy over time
        train_loss_over_time = []
        val_loss_over_time = []
        val_iou_over_time = []
        val_dice_over_time = []
        val_pixel_accuracy_over_time = []

        # Initialize the best IoU vars and lists
        best_iou = 0
        best_dice = 0
        best_pixel_accuracy = 0
        best_iou_over_time = []
        best_dice_over_time = []
        best_pixel_accuracy_over_time = []

        # Start training
        start_time = time.time()

        # Initialize a new run
        self.initialize_new_run("UNet-train-val-run")

        for epoch in range(self.opt.epochs):
            
            # Print the current epoch
            print("Training epoch: {}/{}".format(epoch+1, self.opt.epochs))

            # Train
            model.train()
            avg_loss = 0

            try:
                # Train
                for images, masks, img_paths, mask_paths in tqdm(trainloader):

                    images, masks = images.to(self.device), masks.to(self.device)
                    optimizer.zero_grad() # Set all gradients to 0

                    predictions = model(images) # Feedforward
                    
                    loss=l_bce(predictions, masks) # Calculate the error of the current batch
                    avg_loss+=loss.cpu().detach().numpy()
                    loss.backward() # Calculate gradients with backpropagation
                    optimizer.step() # optimize weights for the next batch

                    # Log the results
                    wandb.log({"train/loss": loss})
                
                avg_loss = avg_loss/len(train_data)
                print("Epoch {}: average  train loss: {:.7f}".format(epoch+1, avg_loss))
                train_loss_over_time.append(avg_loss)

                # Wandb draw loss over time
                wandb.log({"train/avg_loss": avg_loss})
                wandb.log({"train/loss_ot": train_loss_over_time})

                # Validation
                model.eval()
                iou = []
                dice = []
                pixel_accuracy = []
                avg_val_loss = []

                with torch.no_grad():
                    for images, masks, img_paths, mask_paths in valloader:
                        images, masks = images.to(self.device), masks.to(self.device)

                        # Feedforward, softmax
                        predictions = model(images)
                        predicted_masks_bin = torch.sigmoid(predictions) > 0.5

                        # Calculate IoU for validation data
                        iou_score, dice_score, pixel_acc = ModelControler.calculate_iou_f1(predicted_masks_bin, masks)
                        iou_score, dice_score, pixel_acc = iou_score.cpu().numpy(), dice_score.cpu().numpy(), pixel_acc.cpu().numpy()

                        iou.append(iou_score)
                        dice.append(dice_score)
                        pixel_accuracy.append(pixel_acc)

                        avg_val_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                        # Log the results
                        wandb.log({"val/loss": avg_val_loss[-1], "val/iou": iou[-1], "val/dice": dice[-1], "val/pixel_accuracy": pixel_accuracy[-1]})

                # Calculate the average IoU, dice, pixel accuracy and loss over the validation set
                iou = np.mean(iou)
                dice = np.mean(dice)
                pixel_accuracy = np.mean(pixel_accuracy)
                avg_val_loss = np.mean(avg_val_loss)
                # Append the results to the lists
                val_loss_over_time.append(avg_val_loss)
                val_iou_over_time.append(iou)
                val_dice_over_time.append(dice)
                val_pixel_accuracy_over_time.append(pixel_accuracy)

                # Wandb log loss, IoU, Dice and Pixel Accuracy over time
                wandb.log({"val/avg_loss": avg_val_loss})
                wandb.log({"val/avg_iou": iou})
                wandb.log({"val/avg_dice": dice})
                wandb.log({"val/avg_pixel_accuracy": pixel_accuracy})
                wandb.log({ "val/loss_ot": val_loss_over_time })
                wandb.log({ "val/iou_ot": val_iou_over_time })
                wandb.log({ "val/dice_ot": val_dice_over_time })
                wandb.log({ "val/pixel_accuracy_ot": val_pixel_accuracy_over_time })
                
                # Save network weights when the accuracy is great than the best_acc
                if iou > best_iou:
                    print(f"New best found. Current best IoU: {iou}")
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './segmentation/output/UNet_IoU_weights.pth')
                    best_iou_over_time.append(iou)
                    best_iou = iou

                    # Wandb draw best IoU over time
                    wandb.log({'val/best_iou': best_iou_over_time[-1]})

                if dice > best_dice:
                    print(f"New best found. Current best Dice: {dice}")
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './segmentation/output/UNet_Dice_weights.pth')
                    best_dice_over_time.append(dice)
                    best_dice = dice

                    # Wandb draw best Dice over time
                    wandb.log({'val/best_dice': best_dice_over_time[-1]})

                if pixel_accuracy > best_pixel_accuracy:
                    print(f"New best found. Current best Pixel Accuracy: {pixel_accuracy}")
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './segmentation/output/UNet_Pixel_Accuracy_weights.pth')
                    best_pixel_accuracy_over_time.append(pixel_accuracy)
                    best_pixel_accuracy = pixel_accuracy

                    # Wandb draw best Pixel Accuracy over time
                    wandb.log({'val/best_pixel_accuracy': best_pixel_accuracy_over_time[-1]})

                # Print the results
                print("Average IOU: {:.7f}     Best IOU: {:.7f}     Average Dice: {:.7f}     Best Dice: {:.7f}     Average Pixel Accuracy: {:.7f}     Best Pixel Accuracy: {:.7f}    Average Validation Loss: {:.7f}".format(iou, best_iou, dice, best_dice, pixel_accuracy, best_pixel_accuracy, avg_val_loss))

                # Log the results
                with open(logfile, 'a') as file:
                    row = f'Epoch {epoch}: Train loss: {avg_loss}, val loss: {avg_val_loss}, val iou: {iou}, val dice: {dice}, val pixel accuracy: {pixel_accuracy}\n'
                    file.write(row)

            except KeyboardInterrupt:
                break

        # Finish the run
        wandb.finish()

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
        plt.savefig(os.path.join('./segmentation/output', filename))
        
        # Plot equal error rate over time
        plt.figure(figsize=(15, 10))
        plt.plot(range(len(val_iou_over_time[1:])), val_iou_over_time[1:], c="dodgerblue")
        plt.title("IoU per epoch", fontsize=18)
        plt.xlabel("epoch", fontsize=18)
        # plt.ylabel("EER, AUC", fontsize=18)
        plt.legend(['IoU'], fontsize=18)
        filename = f'iou.svg'
        plt.savefig( os.path.join('./segmentation/output', filename))

    # Function to test the model
    def test(self):
        # Initialize the test dataset
        test_data = DatasetFolder(root=os.path.join(self.opt.datasetRoot, 'test-seg'),
                                    image_only_transform=self.image_only_transform_test,
                                    transform=self.image_and_mask_transform_test)

        print(f"Test dataset stats: number of images: {len(test_data)}")

        testloader = DataLoader(test_data, self.opt.batchsize, shuffle=False)

        # Load the CNN model, loss_function and weights
        model = UNet().to(self.device)
        l_bce = nn.BCEWithLogitsLoss()

        # Load the best weights
        if self.opt.best_weights == 'IoU':
            model.load_state_dict(torch.load('./segmentation/output/UNet_IoU_weights.pth')['state_dict'])
        elif self.opt.best_weights == 'Dice':
            model.load_state_dict(torch.load('./segmentation/output/UNet_Dice_weights.pth')['state_dict'])
        elif self.opt.best_weights == 'Pixel_Accuracy':
            model.load_state_dict(torch.load('./segmentation/output/UNet_Pixel_Accuracy_weights.pth')['state_dict'])
        else:
            raise ValueError("Invalid best weights")

        # Start testing
        model.eval()

        # Initialize the lists to store the loss, accuracy and predictions
        iou = []
        dice = []
        pixel_accuracy = []
        avg_test_loss = []

        # Start testing
        start_time = time.time()

        # Initialize a new run
        self.initialize_new_run("UNet-test-run")

        with torch.no_grad():
            for images, masks, _, _ in self.testloader:
                images, masks = images.to(self.device), masks.to(self.device)

                # Feedforward, softmax
                predictions = model(images)

                predicted_masks_bin = torch.sigmoid(images) > 0.5

                # Calculate IoU, Dice score, Pixel Accuracy and loss for test data
                iou_score, dice_score, pixel_acc = ModelControler.calculate_iou_f1(predicted_masks_bin, masks)
                iou_score, dice_score, pixel_acc = iou_score.cpu().numpy(), dice_score.cpu().numpy(), pixel_acc.cpu().numpy()

                iou.append(iou_score)
                dice.append(dice_score)
                pixel_accuracy.append(pixel_acc)
                avg_test_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                # Log the results
                wandb.log({"test/loss": avg_test_loss[-1], "test/iou": iou[-1], "test/dice": dice[-1], "test/pixel_accuracy": pixel_accuracy[-1]})

        # Finish the run
        wandb.finish()

        # Draw IoU, Dice and Pixel Accuracy over time
        wandb.log({ "test/loss_ot": avg_test_loss })
        wandb.log({ "test/iou_ot": iou })
        wandb.log({ "test/dice_ot": dice })
        wandb.log({ "test/pixel_accuracy_ot": pixel_accuracy })

        # Print total testing time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total testing time: {elapsed_time}")

        # Calculate the average IoU and loss over the test set
        iou = np.mean(iou)
        dice = np.mean(dice)
        pixel_accuracy = np.mean(pixel_accuracy)
        avg_test_loss = np.mean(avg_test_loss)

        print("Average Test IOU: {:.7f}     Average Test Dice: {:.7f}     Average Test Pixel Accuracy: {:.7f}     Average Test Loss: {:.7f}".format(iou, dice, pixel_accuracy, avg_test_loss))
        

# MAIN
if __name__ == "__main__":
    # Arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset control
    options.add_argument('--prepare', action='store_true', help='Prepare the dataset.')
    options.add_argument('--resize-prepared', action='store_true', help='Resize the images and masks.')
    options.add_argument('--resize-prepared-preserve-aspect-ratio', action='store_true', help='Resize the images and masks while preserving aspect ratio.')
    options.add_argument('--resize-prepared-size', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    # Model control
    options.add_argument('--train', action='store_true', help='Train the model.')
    options.add_argument('--best-weights', type=str, default='IoU', help='Which weights to use for testing the model: "IoU", "Dice" or "Pixel_Accuracy')
    options.add_argument('--test', action='store_true', help="Test the model.")
    # Configuration
    options.add_argument('--datasetRoot', type=str, default='/app/container/dataset', help='Root directory of the dataset')
    options.add_argument('--trainTestRatio', type=float, default=0.8, help='Ratio of the dataset to be used for training')
    options.add_argument('--trainValRatio', type=float, default=0.5, help='Ratio of the training dataset to be used for validation')
    options.add_argument('--batchsize', type=int, default=4, help='Batch size')
    options.add_argument('--imagesize', type=lambda x: tuple(map(int, x.split(','))), default=(512,384), help='Size of the image (height, width)')
    options.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    opt = options.parse_args()

    # Initialize the model controler
    model_controler = ModelControler(opt)

    # Prepare the dataset
    if opt.prepare:
        print("Preparing the segmentation dataset...")
        model_controler.prepare_dataset_folder()

    # Train the model
    if opt.train:
        print("Training the segmentation model...")
        model_controler.train()
    
    # Test the model
    if opt.test:
        print("Testing the segmentation model...")
        model_controler.test()