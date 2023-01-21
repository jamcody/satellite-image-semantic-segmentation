# SATELLITE IMAGE SEMANTIC SEGMENTATION
This project is a part of a Kaggle cometition in Deep Learning course at CentraleSupelec and is carried out with my team member i.e. Vanshika Sharma
## Project description
The task was to employ deep learning to segment and classify the objects (26 labels) in the image. The dataset comprises 374 satellite images (3x3000x4000) of residential neighborhoods in Houston after the hurricane Harvey. 299 out of the total images have corresponding masks for training deep learning. The competition requires the submission of masks for the remaining 75 images. The evaluation metric is dice score. A snapshot of the the image and provided mask is as followed:

![image](https://user-images.githubusercontent.com/85484281/213866548-7f770301-9f7a-466f-bfee-c6e3004aee18.png)
## Pre-processing and exploration
Key findings after exploration include:
- Unbalanced labels: we oversampled by creating additional 500 images (spliting the image with under-represented labels into 4)
![image](https://user-images.githubusercontent.com/85484281/213866790-ceee17e1-1b0a-439f-bb58-a5a88237b0a7.png)

- Huge image size (3x300x400): we reduced the size to 3x352x352 to speed up the training process

Besides, data augmentation (flipping, corlor adjusting) are also used to improve the accuracy of the model

## Modelling


## Result
