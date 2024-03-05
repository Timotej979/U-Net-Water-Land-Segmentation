# U-Net-Water-Land-Segmentation

![Header](./docs/assets/header.png)

## Abstract

Recent advancements in image segmentation methodologies have significantly impacted the navigational capabilities of unmanned surface vehicles (USVs) over maritime environments. This research delves into the formulation and performance assessment of a U-Net [7] based segmentation model trained on a specialized dataset, modeled after
the MaSTr1325 and MODD2 datasets [2] [3] but recorded on a river surface instead of the sea. The WaSR network, renowned for water segmentation and refinement in maritime obstacle detection [1] [10], was employed to extract masks from the recorded frames. Subsequently, the three-segment masks were transformed into binary masks, distinguishing water from the surroundings, forming the basis for training the U-Net segmentation model. The U-Net preformed exemplary, achieving an Intersection over Union
(IoU) score of 99.08%, a dice score of 99.73%, and pixel accuracy of 99.39%. 

Building upon the successful implementation of the U-Net segmentation model, this study further explores the potential application of the YOLOv8 [4] object detection algorithm for identifying obstacles in the navigational path of unmanned surface vehicles in river environments. Despite the initial promise of YOLOv8, its integration led to less-than-ideal outcomes, with a notable decrease in performance metrics when applied without pre-training on the dataset. Specifically, the Intersection over Union (IoU) score plummeted to 0.01%, and the dice score to 0.03%, starkly contrasting the exceptional results achieved in the segmentation task. This emphasizes the critical importance of proper fine-tuning of YOLOv8 model on domain-specific data to optimize its performance in obstacle detection scenarios. 

The overall results show unique challenges regarding segmentation and detection in river based environments.


## Runing the code

First setup your own [Weights and Biases](https://wandb.ai/site) account settings in the __.env_example__ file and rename it to __.env__ like for example:

```bash
#!/bin/bash

# Set the AI key for wandb
export WANDB_API_KEY='BMif1IPmyg8ceZGLtj3MKN3CnSQ577Pi5w6bRDD2tVIDol9OuVChkiGhNqZtmFoavgGmHLees71RMCyIqGOX2BGiLTnBQH0fQV1ZWIjd7af1KewEC5SDgWhtPaJQg2c0'
export WANDB_ENTITY='JohnDoe123'
```

After this you can edit the __runSegDet.sh__ file in the **/container** folder to enable/disable different commands/features of this project. Bellow is an example of running all possible training and testing commands.

```bash
#!/bin/bash

# Print the WAND environment variables for debugging
printenv | grep WANDB

# Prepare the sub-datasets for segmentation and detection
python3 /app/container/segmentation/run.py --prepare --resize-prepared --resize-prepared-size 512,384
python3 /app/container/detection/run.py --prepare --resize-prepared --resize-prepared-size 512,384 --autolabel --autolabel-method rawcontours
python3 /app/container/detection/run.py --prepare --resize-prepared --resize-prepared-size 512,384 --autolabel --autolabel-method autodistil

# Run segmentation
python3 /app/container/segmentation/run.py --train
python3 /app/container/segmentation/run.py --test --best-weights IoU
python3 /app/container/segmentation/run.py --test --best-weights Dice
python3 /app/container/segmentation/run.py --test --best-weights Pixel_Accuracy

# Run detection
python3 /app/container/detection/run.py --test --test-method pretrained

# The following two methods are still in development
#python3 /app/container/detection/run.py --test --test-method contours
#python3 /app/container/detection/run.py --test --test-method autodistil
```

Before running the code you might need to download the default YOLOv8 weights and place them in the **container/detection/weights** folder for the run script of the detection script to read and use them.


## Project structure

The general project structure is split in the root folder named **/container** from which we have access to 3 main components as folows:

- **/dataset folder** - Contains all raw data (__/RGB__ and __/WASR__ folders) and subseqent datasets used in segmentation (__*-seg__ format folders) and detection (__*-det__ format folders)
- **/segmentation folder** - Contains the code used to run the segmentation U-Net on the dataset with pre-preparation of the data
- **/detection folder** - Contains the code used to run the detection using YOLOv8 on the dataset with pre-preparation of the data


## Introduction

Image segmentation and detection, fundamental tasks in computer vision, play a pivotal role in enhancing the navigational capabilities of unmanned surface vehicles (From
now on USV/USVs) over diverse environments. Recent advancements in this domain have demonstrated promising outcomes, providing a basis for further exploration and refinement.

This study focuses on the development and evaluation of a U-Net based segmentation model, a neural network architecture widely recognized for its effectiveness in image segmentation tasks [7]. The model is trained on a specialized dataset crafted in alignment with the MaSTr-1325 and MODD2 datasets [2] [3], with a distinctive emphasis
on river surfaces. While maritime environments have been extensively studied, the unique challenges posed by river navigation necessitate tailored approaches to both maritime and freshwater surfaces for effective obstacle detection and path planning.

The utilization of the water segmentation and refinement network (From now on WaSR) [1] [10], acknowledged for its prowess in water segmentation and refinement in maritime obstacle detection, serves as a crucial component in this research. By extracting masks from recorded frames, the algorithm contributes to the creation of binary masks
that distinguish water from surrounding elements. These masks are then employed to train the U-Net segmentation model, which, as indicated by performance metrics, exhibits
exceptional accuracy.

Building on the success of the segmentation model, the study extends its focus to the integration of the 8th generation You Only Look Once object (From now on YOLOv8)
detection algorithm [4]. The objective is to assess its applicability for identifying obstacles in the path of USVs navigating river environments. However, preliminary results reveal sub-optimal performance, emphasizing the critical role of fine-tuning the YOLOv8 detector on domain specific data. This highlights the necessity for a nuanced understanding of the environmental context and the detection algorithm architecture in optimizing object detection capabilities for USVs.

As such there is also a need for evaluating aforementioned models using correct performance measures. As such we explore which measures fit the criteria for accurate
evaluation of our models.


## Realted work

The related works regarding segmentation and detection algorithms is split into five unique sections. First we explain the structure of two of the new large-scale maritime
semantic segmentation datasets recorded with and meant for small-scale USVs. Then we explain the WaSR [1] [10] algorithm which we used for primary data preprocessing.
Following it we delve into the architecture of both neural networks used for segmentation and detection, U-Net and YOLOv8. Finally we explore the performance measures
that fit our criteria for correctly evaluating our neural networks.

### Water-surface USV datasets

The MaSTr1325 dataset [2] is a coastal maritime dataset comprised of 1325 diverse images, that were picked with regards to various weather conditions and time of day to
ensure dataset variety, primarily designed for semantic segmentation and obstacle detection. The dataset was acquired using a small water-surface based USV recording more
than 50 hours of footage while manually driving around the coastal areas. Since this is a segmentation dataset each image or rather image pixel of the dataset is labeled with one of the four categories: obstacles or environment, water, sky and Unknown category. The dataset also offers a significant advantage in terms that all images are precisely time-synchronized with on-board GPS and IMU measurements, ensuring accurate alignment for comprehensive analysis.






## References

[1] Bovcon, B., & Kristan, M. (2021). WaSR--A Water Segmentation and Refinement Maritime Obstacle Detection Network. IEEE Transactions on Cybernetics

[2] Bovcon, B., Muhovič, J., Perš, J., & Kristan, M. (2019). The MaSTr1325 dataset for training deep USV obstacle detection models. 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)

[3] Bovcon, B., Muhovič, J., Vranac, D., Mozetič, D., Perš, J., & Kristan, M. (2021). MODS -- A USV-oriented object detection and obstacle segmentation benchmark. IEEE Transactions on Intelligent Transportation Systems.
