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
The data set includes 3482 image of documents over 10 classes.
There is a quite imbalance in data where the label *email* is the most popular.
## Data processing
There is no abnormal case in data. But we need construct the data pipeline to load image and its label from hard disk.
The data processing image include these steps:
		1. Read image and label from directory.
		2. Convert Image to tensor
		3. Resize image to size `(224, 224)`
		4. Normalize image data

Dataset is load by batch of 32.

## Implementation
### Construct the model

My model classification include a featurea extraction which is pretrained with large of data such as VGG, Dense or [DiT](https://arxiv.org/pdf/2203.02378.pdf) followed by a single linear layer, whose output size is the number of classes of the dataset.

![model structure](https://github.com/thaihocnguyen-git/tobacco_classification/blob/main/images/model.png)

### Prepare data
Tobacco dataset is downloaded from [kaggle](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg). 20 percent of data is used for validation phase and the rest is used for traning phase.

I made use of `torchvision.datasets.VisionDataset` to create my custom dataset to load and preprocess image.

### Training
In training phase, I used torch.nn.CrossEntropyLoss and torch.optim.Adam respectively as Loss and Optimizer. The model is trainning in 10 epochs with batch size 64.

### Validation
The accuracy is the used metric because this is the multiclass classification and there is no imbalance in test dataset.
The result of model is below:
| Model | Model size | Accuracy  |
|---------|----------|-----------|
| DiT     | 330MB    |           |
| Dense121| 35MB     |           |
| VGG19   | 522 MB   |           |

### Diffculties and solutions
#### 1. Finding the backbone
There's a huge of pretrained computer vision model a public for transfer learning such as Resnet, Densenet, VGG, InceptionNet,... However almost of them are trained in [Imagenet](https://www.image-net.org/) which is the general purpose dataset. The one I was looking for is the model is pretrained with images of printed document. Finally I found out the DiT, the self-supervised pretrained model. The model is trained in the huge printed dataset. Their work was inspired by the [Biet](https://arxiv.org/pdf/2106.08254.pdf), a transformer based model trained with image masked token task whose objective is
>  to recover the original visual tokens based on the corrupted image patches

This pre-trained model is deally suitable to our document classification task.
Beside, I used the Densenet121 and VGG19 pretrained on Imagenet for comparation.
#### 2. Image normalization prerocessing
Firstly, I had no idea about image nomalization and just trained model with this process. The result turn out is very bad, the model had never converged. I found out that image nomalization the critical part of most of compute vision deep learning model. Thank to `torchvision` for providing `transforms.Normalize` to easily use of normalization.
#### 3. Misunderstading about torch.nn.CrossEntropyLoss functionality
The loss function is to compute the difference between prediction and ground truth label to back propagate to update the model state in training phase. In case of `tensorflow` library, the model usually end up with the Softmax function before feed to loss function. However, the `torch.nn.CrossEntropyLoss` in 	`torch` has already do this stuff for us and no need of additional `Softmax` function. I made a mistake in the first try by adding `Softmax` function, the consequence result is that the model accuracy was looked very bad. After removed this function, the problem was resolved.

## Refinement
1. Model overfitting
	The model loss and accuracy is very high in training dataset but go down in test data. This is overfitting! In futher work, I can improve it by adding Dropout layers, use label smoothing for training phase.
2. Data argumentation
    In the pratical application, the provided input image is not alsway in good condition, somtimes they can be rotated, flipped or noise. To make the model robust, adaptive with these data, we can use data argumentation in training.
3. Web application improvement
	 Because the shortage of time, I could only provide very simple of demonstration application. The work will be better if I make web application become well-looked and add some more function likes give top-k labels or make the restful API to serve other application.

## The classification result
As the accuracy in test dataset:
| Model | Model size | Test Accuracy  |Train Accuracy|
|---------|----------|-----------|------------------|
| DiT     | 330MB    |           |
| Dense121| 35MB     |           |
| VGG19   | 522 MB   |           |

our model is a quite good, but not very good. The problem can easily be figured out is overfitting.  

Beside, the DiT model gives the better than the two others. This is reasonale because it is trained in the same domain dataset.

## Conclusion
Document image classification is an interesting and pratical problem. The transfer learning technique is so good by using the knowledge pretrained from the large dataset and transfer to downstream model. This not only help reduce training time, but make the model become robust with better accuracy.
The most diffculty is th overfitting problem. This is a classical problem in machine learning. There are some technique to duel with it that I should research and apply to improve my model such as: label smothing, early stopping, data argumentation, class weights or just use the large dataset.





