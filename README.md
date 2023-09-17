# BrainTumor

**University Project: Fully automatic brain tumor segmentation method based on CNN and VGG16**

In this paper, we present a fully automatic brain tumor segmentation method based on neural networks convolutions (CNNs) and VGG16 model.

The proposed networks are adapted to gliomas, meningiomas and pituitary images depicted in magnetic resonance imaging (MRI).

By their very nature, these tumors can appear anywhere in the brain and have almost any type of shape, size, and contrast.

The dataset used takes into account 4 classes that include the 3 types of cancer with the addition of a fourth class corresponding to non-cancers.

Our models were trained and tested on the same dataset, where in addition, repeated image cleaning was done. The proposed models are a CNN to which the K-Fold CrossValidation is added and a VGG16 with K-Fold CrossValidation attached.

Finally, the results obtained between the models were compared, both with the initial dataset and with datasets without repeated samples.

##
## DATASET

The image dataset used is an optimization of the Kaggle dataset **"Brain Tumor MRI Dataset"** , a combination of 3 datasets and contains 7022 MRI images of the human brain which are classified into 4 classes:

-  glioma
- meningioma
- pituitary
- no tumor.
  
Where the datasets combined to obtain the resultant are:
- figshare 
- SARTAJ dataset
- Br35H
  
The **figshare dataset** contains 3064 contrast-enhanced T1-weighted images with three brain tumor types.

The **SARTAJ dataset** contains 3264 images divided into 4 classes.

The **Br35H dataset** contains 3060 brain MRI images and was used to obtain the non-tumor class.

The optimization was necessary as identical images were found in both the Training and Testing datasets and, therefore, a potential alteration in the subsequent classification results.

The identification and removal of identical images from the Brain Tumor MRI Dataset was achieved with the use of Visipics, a freely licensed software that uses an algorithm capable of detecting repeated and duplicate files.
By eliminating duplicates, the final dataset was obtained containing 5572 images also divided into 4 classes like the previous one.

##
## APPROACH

Our approach to the problem of brain tumor segmentation was through the use of a convolutional neural network (CNN).

The model sequentially processes each 2D image where each pixel is associated with different image modalities. Like most CNN-based segmentation models, our approach predicts the class of a pixel by processing the M × M patch centered on that pixel.
The input X of our CNN model is therefore a 2D M × M patch with different modalities. The main building block used to build a CNN architecture is the convolutional layer. Different layers can be stacked on top of each other forming a hierarchy of functions. Each level can be understood as extracting features from the previous level in the hierarchy to which it is connected. A single convolutional layer takes as input a stack of input planes and produces as output a certain number of output planes or feature maps. Each feature map can be thought of as a topologically organized map of the responses of a particular spatially local nonlinear feature extractor, which learns its parameters, applied identically to each spatial neighborhood of the input planes in a sliding window. In the case of a first convolutional layer, individual input planes correspond to different MRI modes (in typical computer vision applications, individual input planes correspond to red, green, and blue color channels). In subsequent layers, the input planes are typically made up of the feature maps from the previous layer.

<p></p>

### ARCHITECTURE CNN

<img width="379" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/8ad0e9e6-c352-45ea-bfb7-86056a7f4ce2">

<img width="518" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/5c84951a-89fa-487f-90a2-bf402cebae18">

##
### RESULT CNN 



The results obtained from the evaluation of the model were as follows with the relative confusion matrix:

<img width="578" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/5a92cb40-474f-47f4-9188-ea32cf484407">

<p></p>

<img width="578" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/39d47575-02bd-4f84-9396-a5236988b9b6">

<p></p>

**Testing:**

30/30 [==============================] - 2s 58ms/step - loss: 1.5523- accuracy: 0.82
 
<img width="578" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/3259252d-c72a-4117-9af5-b3e690b4ad2c">

<img width="723" alt="Schermata 2023-09-16 alle 11 07 25" src="https://github.com/haxkadc/BrainTumor/assets/134702013/4be0db24-7be6-4da5-8171-55b19d40d9fc">

##
### K-FOLD CROSS VALIDATION

With  **K = 5**, we proceeded to recreate the model with the same structure used for the training and testing phase, same optimizer and same type of loss, and then use the K-Fold cross validation

Inserisci


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/a9b8029a-5a40-4954-b227-59b2ed2c1ab9"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/8f54f8e5-496e-4c29-879d-0dab993a24fc">


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/fd1b9741-9dff-4746-b1eb-907343bd2495"> <img width="413" alt="image" src="http://github.com/haxkadc/BrainTumor/assets/134702013/161f0cbc-63ee-4d56-98d5-f9e705c6cd83">


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/0969b3d9-3594-426b-8494-088036064184"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/d2deb849-0fe7-4345-9fde-3c9fabdc3171">



<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/88c322f7-593b-4023-a0be-0514bc9b4c06"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/4fcb1532-0472-461a-9d2b-b05110185e24">


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/2e60c1ac-ea70-43c8-a97b-2ea8745919b2"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/34b8efd2-91f7-4cc6-aae0-f75810c9eb3e">

 
<img width="700" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/b31a0b55-a443-4ce1-bf2e-ed571bf934dc">

##
### ARCHITECTURE VGG16

A fine tuning was made with VGG16 model of which the upper layers, i.e. the dense ones, were freezato be able to take images of size 128x128 as input.
The weights used are those obtained from pretraining on imagenet.
In addition, the model has been set with false flags and excluded it from subsequent training
(vggmodel.trainable = False).
Then, after the aforementioned layer of VGG16 a Flatten() layer was added, two dense ones both having 50 neurons and a ReLu activation function and finally a last dense layer with 4 neurons.
The structure of the model is as follows:

<p></p>

<img width="520" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/cc6e02cc-464e-47ba-b42c-701fc4dd8177">

<p></p>


##
### RESULT VGG16

<img width="516" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/8f41fc0d-0b64-4790-9afc-c40726ab8414">

<img width="516" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/8768085b-152a-416b-b013-1c0ce0ee6141">

<p></p>

**Testing:**


30/30 [==============================] - 30s 991ms/step - loss: 1.0355 - accuracy: 0.8687


<p></p>

<img width="516" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/9d148fcf-b8d4-4b05-8b7b-92ede164ede6">


<img width="723" alt="Schermata 2023-09-16 alle 11 07 25" src="https://github.com/haxkadc/BrainTumor/assets/134702013/69ec5edb-440b-4736-a9dc-b363be72be50">


##
### K-FOLD CROSS VALIDATION


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/96fb7c4a-7acb-4767-bc6c-6026473b820c"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/4f904fd8-5860-4f07-96dd-d9213ee7ee2f">


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/443ce2a3-0d23-4a08-b7e8-7815ea651ece"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/f6f23190-905a-4b62-9837-a6517b4d515d">



<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/8804d432-4e88-40b8-9e6e-ed155e0d76b5"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/9673a39a-81be-4104-b4a7-22493a91de5a">


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/fe0ce338-2615-4089-a190-44ab9a46187e"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/b90e6b56-385b-4875-b8a1-f2842563e127">


<img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/5df50df6-3d5b-4846-b8ca-11b1159fe053"> <img width="413" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/3dbf6eb2-7cba-4ed6-90e2-7820e6b7f7d2">


<img width="600" alt="image" src="https://github.com/haxkadc/BrainTumor/assets/134702013/1ff1371d-bb36-473f-99e5-01651ef99aad">


