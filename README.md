# Python based Project – Learn to Build Image Caption Generator with CNN & LSTM
video : https://drive.google.com/file/d/18heus65r6tcl37hdp4nBQL8jv0mD2OmC/view?usp=sharing

You saw an image and your brain can easily tell what the image is about, but can a computer tell what the image is representing? Computer vision researchers worked on this a lot and they considered it impossible until now! With the advancement in Deep learning techniques, availability of huge datasets and computer power, we can build models that can generate captions for an image.

This is what we are going to implement in this Python based project where we will use deep learning techniques of Convolutional Neural Networks and a type of Recurrent Neural Network (LSTM) together.

# What is Image Caption Generator?
Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

# The Dataset of Python based Project

For the image caption generator, we will be using the Flickr_8K dataset. There are also other big datasets like Flickr_30K and MSCOCO dataset but it can take weeks just to train the network so we will be using a small Flickr8k dataset. The advantage of a huge dataset is that we can build better models.

# Pre-requisites
This project requires good knowledge of Deep learning, Python, working on Jupyter notebooks, Keras library, Numpy, and Natural language processing.

# Make sure you have installed all the following necessary libraries:

pip install tensorflow
keras
pillow
numpy
tqdm
jupyterlab

# What is CNN?

Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images.
CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc.It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images. It can handle the images that have been translated, rotated, scaled and changes in perspective.

# What is LSTM?
LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

# Image Caption Generator Model

So, to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.

CNN is used for extracting features from the image. We will use the pre-trained model Xception.
LSTM will use the information from CNN to help generate a description of the image.

Project File Structure
Downloaded from dataset:
Flicker8k_Dataset – Dataset folder which contains 8091 images.
Flickr_8k_text – Dataset folder which contains text files and captions of images.

The below files will be created while making the project.
Models – It will contain our trained models.
Descriptions.txt – This text file contains all image names and their captions after preprocessing.
Features.p – Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
Tokenizer.p – Contains tokens mapped with an index value.
Model.png – Visual representation of dimensions of our project.
Testing_caption_generator.py – Python file for generating a caption of any image.
Training_caption_generator.ipynb – Jupyter notebook in which we train and build our image caption generator.


