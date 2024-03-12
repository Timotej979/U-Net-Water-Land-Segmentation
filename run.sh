#!/bin/bash

# Run the environment initialization script
source .env

# Check if the current_dir + /container/detection/weights directory exists
WEIGHTS_DIR="$PWD/container/detection/weights"

if [ ! -d "$WEIGHTS_DIR" ]; then
    # Create the /container/detection/weights directory
    echo "Creating the /container/detection/weights directory"
    mkdir -p $WEIGHTS_DIR
fi

# Check if the directory has yolov8n/s/m/l/x.pt files
if [ ! -f "$WEIGHTS_DIR/yolov8n.pt" ] || [ ! -f "$WEIGHTS_DIR/yolov8s.pt" ] || [ ! -f "$WEIGHTS_DIR/yolov8m.pt" ] || [ ! -f "$WEIGHTS_DIR/yolov8l.pt" ] || [ ! -f "$WEIGHTS_DIR/yolov8x.pt" ]; then
    # Redownload the weights files
    echo "Redownloading the weights files"
    wget -O $WEIGHTS_DIR/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
    wget -O $WEIGHTS_DIR/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
    wget -O $WEIGHTS_DIR/yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt
    wget -O $WEIGHTS_DIR/yolov8l.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt
    wget -O $WEIGHTS_DIR/yolov8x.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt
fi


# Check command arguments for ./run.sh
if [ "$1" == "cpu" ]; then
    # Run the docker-compose file with the CPU version of the docker image
    echo "Running the CPU version of the docker image"
    docker compose -f cpu.yml build
    docker compose -f cpu.yml up

elif [ "$1" == "gpu" ]; then
    # Run the docker-compose file with the GPU version of the docker image
    echo "Running the GPU version of the docker image"
    docker compose -f gpu.yml build
    docker compose -f gpu.yml up

else
    # Print the help message
    echo "Usage: ./run.sh [cpu|gpu]"
    echo "  cpu: Run the docker-compose file with the CPU version of the docker image"
    echo "  gpu: Run the docker-compose file with the GPU version of the docker image"
    echo "  Not specifying an argument will print this help message"
fi