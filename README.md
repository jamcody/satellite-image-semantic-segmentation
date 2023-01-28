# SATELLITE IMAGE SEMANTIC SEGMENTATION
This project is a part of a Kaggle cometition in Deep Learning course at CentraleSupelec
## Project description
The task is to segment the objects (26 labels) in the satellite images. The dataset comprises 374 images (3x3000x4000) of residential neighborhoods in Houston after the hurricane Harvey. 299 out of the total images have corresponding masks (an image where each pixel is assigned 1 out of 26 values corresponding to 26 labels). These are used for training deep learning models. The masks for the remaining 75 images are to be predicted. U-net, PSPNet & Deeplabv3 are employed and evaluated using dice score and cross entrophy loss. Deeplabv3 outperformed with 72.65% accuracy and 0.6 loss. A snapshot of the images and provided masks are as followed:

![image](https://user-images.githubusercontent.com/85484281/213866548-7f770301-9f7a-466f-bfee-c6e3004aee18.png)
## Pre-processing and exploration
After the exploration, we decided to:
- oversample by creating additional 84 images to address the imbalanced labels (spliting the images containing under-represented labels into 4)

![image](https://user-images.githubusercontent.com/85484281/213866790-ceee17e1-1b0a-439f-bb58-a5a88237b0a7.png)

*Label list = {
'Property Roof': 0,
 'Secondary Structure': 1,
 'Swimming Pool': 2,
 'Vehicle': 3,
 'Grass': 4,
 'Trees / Shrubs': 5,
 'Solar Panels': 6,
 'Chimney': 7,
 'Street Light': 8,
 'Window': 9,
 'Satellite Antenna': 10,
 'Garbage Bins': 11,
 'Trampoline': 12,
 'Road / Highway': 13,
 'Under Construction / In Progress Status': 14,
 'Power Lines & Cables': 15,
 'Bridge': 16,
 'Water Tank / Oil Tank': 17,
 'Parking Area - Commercial': 18,
 'Sports Complex / Arena': 19,
 'Industrial Site': 20,
 'Dense Vegetation / Forest': 21,
 'Water Body': 22,
 'Flooded': 23,
 'Boat': 24,
 'Parking Area': 25}*

- reduced the size from (3x3000x4000) to (3x400x400) to speed up the training process

- augmented images (flipping & adjusting color/ brightness, etc.) to improve the accuracy of the model

## Modelling
In this project, I used Resnet101 pre-trained on ImageNet as backbone to help the network to learn the features more effectively, speeding up the training

**1. Unet** is one of the most widely used network for semantic segmentation problem (upgraded version of FCN). The network basically 
- encodes: extract features from the images & reduce image size to save computation time
- decodes: reconstruct the masks using 1) extracted features and 2) information lost during encoding stage (using skip connections)

![image](https://user-images.githubusercontent.com/85484281/215272882-add332f0-afe3-466d-8291-8c343fa54c30.png)

**2. PSPNet** shook the semantic segmentation field by winning the ImageNet Scene Parsing Challenge 2016. PSPNet is designed to mainly address the context confusion. For example, the boat is predicted to be a car because the shape is similar. If the context i.e. water is considered, this will not happen. As such, in PSPNet architecture, many sub-regions (local info) & full-region (global info) of the last cnn layer are used to extract features. They are, then, concatenated to construct the final feature representation.

![image](https://user-images.githubusercontent.com/85484281/215271892-f9784360-9a77-4e63-9d5c-6a7a9eff5714.png)

**3. Deeplabv3**

## Findings

## References
https://arxiv.org/pdf/1505.04597.pdf

https://towardsdatascience.com/u-nets-with-resnet-encoders-and-cross-connections-d8ba94125a2c

https://jiaya.me/papers/PSPNet_cvpr17.pdf

https://developers.arcgis.com/python/guide/how-deeplabv3-works/
