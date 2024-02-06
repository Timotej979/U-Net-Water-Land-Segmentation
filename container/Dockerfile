# Use prebuilt pytorch image as base, change tag as needed if you want to use cuda
# Source: https://github.com/cnstark/pytorch-docker

# Available tags:
#  - 1.7.0-py3.8.13-cuda11.0.3-ubuntu18.04
#  - 1.7.0-py3.8.13-cuda11.0.3-devel-ubuntu18.04
#  - 1.7.0-py3.8.13-ubuntu18.04

FROM cnstark/pytorch:1.7.0-py3.8.13-ubuntu18.04

# Set the timezone non-interactively, othherwise tzdata will ask for input
ARG TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install aptitude dependencies
RUN apt-get -y update && \
    apt-get -y install python3

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y libgl1-mesa-dev cmake python3-pip libopencv-dev git

WORKDIR /app

# Copy the requirements
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip3 install -r requirements.txt && \
    apt-get clean