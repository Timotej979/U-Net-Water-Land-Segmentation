#!/bin/bash

# Run the environment initialization script
source .env

# Run the docker-compose file
docker-compose build
docker-compose up