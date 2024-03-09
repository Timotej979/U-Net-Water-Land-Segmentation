#!/bin/bash

# Print the WAND environment variables for debugging
printenv | grep WANDB

# Prepare the sub-datasets for segmentation and detection
#python3 /app/container/segmentation/run.py --prepare --resize-prepared --resize-prepared-size 512,384
#python3 /app/container/detection/run.py --prepare --resize-prepared --resize-prepared-size 512,384 --autolabel --autolabel-method rawcontours
#python3 /app/container/detection/run.py --prepare --resize-prepared --resize-prepared-size 512,384 --autolabel --autolabel-method autodistil

# Run segmentation
## Training
#python3 /app/container/segmentation/run.py --train
## Testing
#python3 /app/container/segmentation/run.py --test --best-weights IoU
#python3 /app/container/segmentation/run.py --test --best-weights Dice
#python3 /app/container/segmentation/run.py --test --best-weights Pixel_Accuracy

# Run detection
## Training
python3 /app/container/detection/run.py --train --train-method contours
#python3 /app/container/detection/run.py --train --train-method autodistil
## Testing
#python3 /app/container/detection/run.py --test --test-method pretrained
python3 /app/container/detection/run.py --test --test-method contours
#python3 /app/container/detection/run.py --test --test-method autodistil