#!/bin/bash

# Print the WAND environment variables for debugging
printenv | grep WANDB

# Prepare the sub-datasets for segmentation and detection
python3 /app/container/segmentation/run.py --prepare
python3 /app/container/detection/run.py --prepare --autolabel --autolabel-method rawcontours
python3 /app/container/detection/run.py --prepare --autolabel --autolabel-method autodistil

# Run segmentation
## Training
#python3 /app/container/segmentation/run.py --train
## Testing
#python3 /app/container/segmentation/run.py --test --best-weights IoU
#python3 /app/container/segmentation/run.py --test --best-weights Dice
#python3 /app/container/segmentation/run.py --test --best-weights Pixel_Accuracy

# Run detection

## Train test loops for contour based detection
### Nano size
#python3 /app/container/detection/run.py --model-size n --train --train-method contours
#python3 /app/container/detection/run.py --model-size n --test --test-method contours
### Small size
#python3 /app/container/detection/run.py --model-size s --train --train-method contours
#python3 /app/container/detection/run.py --model-size s --test --test-method contours
### Medium size
#python3 /app/container/detection/run.py --model-size m --train --train-method contours
#python3 /app/container/detection/run.py --model-size m --test --test-method contours
### Large size
#python3 /app/container/detection/run.py --model-size l --train --train-method contours
#python3 /app/container/detection/run.py --model-size l --test --test-method contours
### Extra large size
#python3 /app/container/detection/run.py --model-size x --train --train-method contours
#python3 /app/container/detection/run.py --model-size x --test --test-method contours

## Train test loops for autodistil based detection
## Nano size
#python3 /app/container/detection/run.py --model-size n --train --train-method autodistil
#python3 /app/container/detection/run.py --model-size n --test --test-method autodistil
## Small size
#python3 /app/container/detection/run.py --model-size s --train --train-method autodistil
#python3 /app/container/detection/run.py --model-size s --test --test-method autodistil
## Medium size
#python3 /app/container/detection/run.py --model-size m --train --train-method autodistil
#python3 /app/container/detection/run.py --model-size m --test --test-method autodistil
## Large size
#python3 /app/container/detection/run.py --model-size l --train --train-method autodistil
#python3 /app/container/detection/run.py --model-size l --test --test-method autodistil
## Extra large size
#python3 /app/container/detection/run.py --model-size x --train --train-method autodistil
#python3 /app/container/detection/run.py --model-size x --test --test-method autodistil