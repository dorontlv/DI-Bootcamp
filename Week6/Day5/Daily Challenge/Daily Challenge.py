'''


Image Classification ...


Daily Challenge: Image Classification with Data Augmentation (Cats vs Dogs)


What You’ll learn
How to preprocess image data for a convolutional neural network (CNN).
How to apply data augmentation techniques to improve model generalization.
How to build and train a CNN for binary image classification.
How to use dropout to reduce overfitting in a neural network.


What you will create
A binary image classification model to distinguish between images of cats and dogs.
A visualization of training and validation metrics to analyze model performance.


What You Need to Do

'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

1. Preprocess the Data:

Download and extract the Cats vs Dogs dataset here.
Use ImageDataGenerator to rescale and augment the training images (e.g., horizontal flip, rotation, zoom, and shifts).
Create separate generators for training and validation data.

'''

import os

# Define paths
base_dir = 'Dogs vs Cats'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Define image dimensions and batch size
IMG_HEIGHT, IMG_WIDTH = 256, 256
batch_size = 32

# ~~~~~~~~~~~~~~~~~


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Data augmentation for training data
image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5,
    validation_split=0.2  # Split for validation
)

# Create separate generators for training and validation data.
# You can't create 2 different generators for the same data, because on the first generator we split the data into training an validation.
# So I will use only one generator for both training and validation data.

# ~~~~~~~~~~~~~~~~~

# Training data generator
train_data_gen = image_gen_train.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)



# Validation data generator
val_data_gen = image_gen_train.flow_from_directory(
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


2. Build the Model:

Create a CNN with:
Three convolutional layers with ReLU activation and max-pooling.
Dropout layers to reduce overfitting.
A fully connected layer with 512 units and ReLU activation.
An output layer with a single unit and sigmoid activation for binary classification.
Compile the model using the Adam optimizer and binary cross-entropy loss.

'''

# Build the CNN model
model = Sequential([
    
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),  # 3 colors
    MaxPooling2D(pool_size=(2, 2)),
    
    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Third convolutional layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten the feature maps
    Flatten(),
    
    # Fully connected layer with dropout
    Dense(512, activation='relu'),
    Dropout(0.5),
    
    # Output layer for binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


3. Train the Model:

Train the model on the augmented training data for 15 epochs.
Use the validation data to monitor performance during training.

'''

# Train the model
history = model.fit(
    train_data_gen,
    epochs=15,
    validation_data=val_data_gen
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
4. Evaluate the Model:

Plot the training and validation accuracy and loss to detect overfitting.
Analyze the impact of data augmentation and dropout on model performance.
'''

import matplotlib.pyplot as plt

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''


5. Bonus:

Retry training the model after Data augmentation: Data augmentation takes the approach of generating more training data from existing training samples by augmenting the samples using random transformations that yield believable-looking images. The goal is the model will never see the exact same picture twice during training. This helps expose the model to more aspects of the data and generalize better.
There is multiple methods to augment data:

Apply horizontal flip
Randomly rotate the image
Apply zoom augmentation
Here the code for Data Augmentation

'''

