#!/bin/bash

# Run the environment initialization script
source .env


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