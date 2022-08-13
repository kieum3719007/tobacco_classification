# Document Classification

## Project Overview

Document image understanding or document image information retrieval has a wide range application. Getting structure information from printed or handwritting document automatically helps reduce much of time and effor in many tasks. Deeplearning and computer vision is getting more and more success with many robust model achieving excellent accuracy. Document classification is one of important proccesses in image document understanding. The more accuracy of classification the better preparing for futher task such as optical character recognizing or information retrieval. 

In this project, I explored and deal with the [Tobacco3482 Dataset](https://wiki.umiacs.umd.edu/clip/index.php/Main_Page), a document classification dataset. As many pretrained model are public for downstream tasks recent years ago, I will apply transfer learning to this classfication task.
## Problem statement
1. Download and explore the **Tobacco3428** dataset.
2. Split train/test and load dataset
3. Preprocess images:
a. Convert image to tensor.
b. Resize image to `224 x 224 x 3`
c. Normalize image with imagenet's mean and standart deviation.
4. Create model:
a. Load pretrained model as feature extractor. I choosen three model as my backbone: VGG19, Dense121 and DiT.
b. Create classification head with pytorch deep learning library for our dataset.
5. Train models with **Tobacco3428** dataset.
6. Validate and compare the result of models.
7. Create a simple web application for classification demostration.
## Metric
The *accuracy* score is used to evaluate the metric because we treat each class has the same importance and our dataset is not too imbalance.
## Data Exploration

