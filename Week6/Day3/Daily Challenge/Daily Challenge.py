'''

Classifying Handwritten Digits...
Last Updated: March 10th, 2025

Daily Challenge: Classifying Handwritten Digits with CNNs


👩‍🏫 👩🏿‍🏫 What You’ll learn
How to load and preprocess the MNIST dataset.
How to build a basic Fully Connected Neural Network for image classification.
How to build and train a Convolutional Neural Network (CNN) for image classification.
Understanding the impact of different network architectures on performance.
Basic Keras functionalities for model building and training.


🛠️ What you will create
You will create two models:

A Fully Connected Neural Network (Dense layers) to classify handwritten digits from the MNIST dataset.
A Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset, and compare its performance with the first model.


What You Need to Do
Load the MNIST dataset:

Use keras.datasets.mnist.load_data() to load the training and testing data.
Print the shapes of the loaded data to understand the dataset structure.
Preprocess the data for a Fully Connected Neural Network:

Flatten the images from 28x28 to 784 pixels.
Normalize the pixel values by dividing by 255.
One-hot encode the target labels using keras.utils.np_utils.to_categorical().
Build and train a Fully Connected Neural Network:

Create a Sequential model.
Add Dense layers with appropriate activation functions (e.g., ReLU and softmax).
Compile the model with an optimizer (e.g., Adam), loss function (e.g., categorical cross-entropy), and metrics (e.g., accuracy).
Train the model using model.fit() and evaluate its performance.
Preprocess the data for a Convolutional Neural Network:

Reshape the input data to the shape expected by a Conv2D layer (e.g., (60000, 28, 28, 1)).
Normalize the pixel values by dividing by 255.
One-hot encode the target labels using keras.utils.np_utils.to_categorical().
Build and train a Convolutional Neural Network:

Create a Sequential model.
Add Conv2D and MaxPool2D layers.
Add a Flatten layer.
Add Dense layers with appropriate activation functions.
Compile and train the model, similar to the Fully Connected Neural Network.
Compare the performance:

Analyze the accuracy of both models.
Observe the difference between the Fully connected model and the CNN model.
Push your code to GitHub.


'''







# ****************************************************************************************************
# 
# This question is exactly the same question as the one on the exercises, so I won't do it again.
# 
# ****************************************************************************************************



