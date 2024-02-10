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

        # Define the transformations
        self.define_transformations()

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
        if self.opt.resize_prepaired:
            self.dataset_preparation_class.resize_images_and_masks(self.opt.resize_prepaired_size, self.opt.resize_prepaired_preserve_aspect_ratio)

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

        # Initialize the best IoU list
        best_iou_over_time = []

        # Start training
        start_time = time.time()

        for epoch in range(self.opt.epochs):
            # Initialize a new run
            self.initialize_new_run(f"train-val-run-{epoch+1}")
            
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
                data = [{"Epoch": x, "Train Loss": y} for x, y in enumerate(train_loss_over_time)]
                table = wandb.Table(data=data, columns=["Epoch", "Train Loss"])
                wandb.log(
                    {
                        "train_loss": wandb.plot.line(
                            table, "Epoch", "Train Loss", title="Train Loss over time"
                        )
                    }
                )


                # Validation
                model.eval()
                iou = []
                avg_val_loss = []

                with torch.no_grad():
                    for images, masks, img_paths, mask_paths in valloader:
                        images, masks = images.to(self.device), masks.to(self.device)

                        # Feedforward, softmax
                        predictions = model(images)
                        predicted_masks_bin = torch.sigmoid(predictions) > 0.5

                        # Calculate IoU for validation data
                        iou.append(self.calculate_iou(predicted_masks_bin, masks).cpu().numpy())

                        avg_val_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                        # Log the results
                        wandb.log({"val/loss": avg_val_loss[-1], "val/iou": iou[-1]})

                # Calculate the average IoU and loss over the validation set
                iou = np.mean(iou) 
                avg_val_loss = np.mean(avg_val_loss)
                val_iou_over_time.append(iou)
                val_loss_over_time.append(avg_val_loss)

                # Save network weights when the accuracy is great than the best_acc
                if iou > best_iou:
                    print(f"New best found. Current best IoU: {iou}")
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './segmentation/output/UNet_weights.pth')
                    best_iou_over_time.append(iou)

                    # Wandb draw best IoU over time
                    data = [{"Epoch": x, "Best IoU": y} for x, y in enumerate(best_iou_over_time)]
                    table = wandb.Table(data=data, columns=["Epoch", "Best IoU"])
                    wandb.log({ "best_iou": wandb.plot.line( table, "Epoch", "Best IoU", title="Best IoU over time" ) })

                # Print the results
                print("Average IOU: {:.7f}     Best IOU: {:.7f}, val loss: {:.7f}".format(iou, best_iou_over_time[-1], avg_val_loss))

                # Wandb draw loss over time
                data = [{"Epoch": x, "Val Loss": y} for x, y in enumerate(val_loss_over_time)]
                table = wandb.Table(data=data, columns=["Epoch", "Val Loss"])
                wandb.log({ "val_loss": wandb.plot.line( table, "Epoch", "Val Loss", title="Val Loss over time" ) })

                # Wandb draw IoU over time
                data = [{"Epoch": x, "Val IoU": y} for x, y in enumerate(val_iou_over_time)]
                table = wandb.Table(data=data, columns=["Epoch", "Val IoU"])
                wandb.log({ "val_iou": wandb.plot.line( table, "Epoch", "Val IoU", title="Val IoU over time" ) })

                # Log the results
                with open(logfile, 'a') as file:
                    row = f'Epoch {epoch}: Train loss: {avg_loss}, val loss: {avg_val_loss}, val iou: {iou}' '\n'
                    file.write(row)

            except KeyboardInterrupt:
                break

            finally:
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
        model.load_state_dict(torch.load('./segmentation/output/UNet_weights.pth')['state_dict'])
        model.eval()

        # Initialize the lists to store the loss, accuracy and predictions
        iou = []
        avg_test_loss = []
        all_labels = []
        all_predictions = []

        # Start testing
        start_time = time.time()

        # Initialize a new run
        self.initialize_new_run("test-run")

        with torch.no_grad():
            for images, masks, _, _ in self.testloader:
                images, masks = images.to(self.device), masks.to(self.device)

                # Feedforward, softmax
                predictions = model(images)

                predicted_masks_bin = torch.sigmoid(images) > 0.5
                
                # Flatten the masks and predictions
                all_labels.extend(masks.cpu().numpy().flatten())
                all_predictions.extend(predicted_masks_bin.cpu().numpy().flatten())

                # Calculate IoU and loss for test data
                iou.append(self.calculate_iou(predicted_masks_bin, masks).cpu().numpy())
                avg_test_loss.append(l_bce(predictions, masks).cpu().detach().numpy())

                # Log the results
                wandb.log({"test/loss": avg_test_loss[-1], "test/iou": iou[-1]})

        # Finish the run
        wandb.finish()

        # Print total testing time
        end_time = time.time()
        elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
        print(f"Total testing time: {elapsed_time}")

        # Calculate the average IoU and loss over the test set
        iou = np.mean(iou)
        avg_test_loss = np.mean(avg_test_loss)

        print("Average Test IOU: {:.7f}, Test loss: {:.7f}".format(iou, avg_test_loss))

        # Plot the ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(15, 10))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title('Receiver Operating Characteristic', fontsize=18)
        plt.legend(loc="lower right", fontsize=18)
        filename = f'roc.svg'
        plt.savefig(os.path.join('./segmentation/output', filename))

        # Plot ROC to wandb
        wandb.log({"roc": wandb.plot.roc_curve(all_labels, all_predictions, labels=["Segmented pixels"], title="ROC for water/land segmentation")})

        # Print the classification report
        print("Classification report:")
        print(classification_report(all_labels, all_predictions))




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

    