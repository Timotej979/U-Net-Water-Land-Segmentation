#!/bin/bash

# Run the environment initialization script
source .env

# Set the dataset and output directory
DATASET_DIR=$(pwd)/dataset
OUTPUT_DIR=$(pwd)/output

# Build the docker image
docker build -t unet-image .

# Run the application
docker run -it -v $DATASET_DIR:/dataset:rw -v $DETECTOR_OUTPUT_DIR:/output unet-image /bin/bash