#!/bin/bash

# Print the WAND environment variables for debugging
printenv | grep WAND

# Run segmentation
python3 /app/container/segmentation/run.py --prepare --resize-prepaired --resize-prepaired-size 512,384 --train --test