'''

Exercises XP

XP Exercises: Flower Classification using CNN

Building a CNN for multi-class image classification.
Data loading and preprocessing using image_dataset_from_directory.
Image visualization techniques.
Model architecture design, compilation, and training.
Evaluating model performance using accuracy and loss plots.

What you will create
A CNN model capable of classifying images of 14 different flower species with high accuracy.

Dataset : Flower Classification Dataset Description
This dataset is designed for multi-class image classification of 14 different flower species. The goal is to train a model that can accurately categorize flower images into their respective species.

Dataset Overview
Number of Classes: 14
Flower Species:
Astilbe
Bellflower
Black-eyed Susan
Calendula
California Poppy
Carnation
Common Daisy
Coreopsis
Dandelion
Iris
Rose
Sunflower
Tulip
Water Lily
Data Organization: Images are organized into a directory structure, with likely separate folders for each species. Training and validation splits are provided.
Image Format: Likely JPEG or PNG. The provided code initially assumes RGB (3-channel) images.
Image Size: Images are resized to 256x256 pixels in the provided code. Original sizes may vary.
Dataset Size:
Training Set: 13,642 images
Validation Set: 98 images

'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''

🌟 Exercise 1: Data Exploration and Visualization
1. Objective: Familiarize yourself with the dataset.
2. Task:
* Load the dataset using image_dataset_from_directory.
* Print the number of images per class.
* Modify the visualize_images function to display a grid of 3x3 images for each flower class. Ensure the class name is displayed as the title for each grid.
* Analyze the images. What are some challenges you anticipate in classifying these flowers? (e.g., similar colors, shapes, variations within a species).

'''

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ~~~~~~~~~~~~~~~~~~~~

# Load the dataset
train_dataset = image_dataset_from_directory(
    './Data - Flower Classification/train',
    seed=123,
    image_size=(256, 256)
)

validation_dataset = image_dataset_from_directory(
    './Data - Flower Classification/val',
    seed=123,
    image_size=(256, 256)
)

# ~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_images(images, the_class_name):
    
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)  # Create a 3x3 grid
        the_image, the_label = images[i]
        plt.imshow(the_image.numpy().astype("uint8"))
        if i == 0:
            plt.suptitle(the_class_name, fontsize=16)
        plt.axis('off')  # Hide axes for better visual clarity
    plt.show()
    

# ~~~~~~~~~~~~~~~~~~~~

# Print the number of images per class

class_names = train_dataset.class_names
print("Class names:", class_names)

# we unbatch the dataset, then we filter only the images of a certain class name. (there are 13000 images - so it takes time)
for class_name in class_names:
    list_of_images_in_class = list(train_dataset.unbatch().filter(lambda x, y: y == class_names.index(class_name)))
    count = len(list_of_images_in_class)
    print(f"Number of images in class {class_name}: {count}")
    
    visualize_images(list_of_images_in_class, class_name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''

🌟 Exercise 2: Model Architecture Design
1. Objective: Design a CNN architecture suitable for this task.
2. Task:
Start with the provided model architecture.
Experiment with the number of convolutional layers, filters, kernel sizes, and max-pooling layers.
Try different combinations of dense layers and dropout rates.
Consider adding Batch Normalization layers after convolutional or dense layers.
Justify your architectural choices. Why did you choose these specific layers and parameters?

'''

from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from tensorflow.keras import layers, models

# Create the CNN model
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # First Convolutional Block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Block
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# Define the input shape and the number of classes
input_shape = (256, 256, 3)  # 256x256 RGB images
num_classes = 14  # 14 flower species

# Build the model
model = build_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model summary
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''


🌟 Exercise 3: Hyperparameter Tuning
1. Objective: Optimize the model’s performance by tuning hyperparameters.
2. Task:

Experiment with different optimizers (e.g., Adam, RMSprop, SGD).
Vary the learning rate and batch size.
Try different loss functions (if applicable).
Use techniques like learning rate scheduling or early stopping to improve training.
Keep track of your experiments and their results. Which combination of hyperparameters yielded the best performance?


'''

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Experiment with different optimizers
optimizers = {
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
}

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Experiment with different batch sizes
batch_sizes = [16, 32, 64]

# Results dictionary to store performance metrics
results = {}

import copy

for optimizer_name, _ in optimizers.items():
    for batch_size in batch_sizes:
        
        print(f"Training with optimizer: {optimizer_name}, batch size: {batch_size}")
        
        # Build and compile the model
        model = None
        model = build_cnn_model(input_shape, num_classes)

        optimizer = copy.deepcopy(optimizers[optimizer_name])  # Get a completely new copy (deepcopy) of the optimizer from the dictionary

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=10,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
            batch_size=batch_size
        )
        
        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(validation_dataset)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        
        # Store the results in a table
        results[(optimizer_name, batch_size)] = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'history': history.history
        }

# Print the best combination of hyperparameters
best_combination = max(results, key=lambda x: results[x]['val_accuracy'])
print(f"Best combination: Optimizer={best_combination[0]}, Batch Size={best_combination[1]}")
print(f"Validation Accuracy: {results[best_combination]['val_accuracy']}")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''

🌟 Exercise 4: Data Augmentation
1. Objective: Improve model generalization by applying data augmentation.
2. Task:

Implement data augmentation using ImageDataGenerator.
Explore different augmentation techniques:
Rotation
Flipping (horizontal/vertical)
Zooming
Shifting (width/height)
Shearing
Determine which augmentations are most effective for this dataset and explain why.

'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation techniques
data_augmentation = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)


import numpy as np

# Apply data augmentation to the training dataset
# Convert the train_dataset to NumPy arrays for compatibility with ImageDataGenerator
train_images = []
train_labels = []

for images, labels in train_dataset.take(1):  # Take only one batch for demonstration
    # Convert images and labels to NumPy arrays
    train_images.append(images.numpy())
    train_labels.append(labels.numpy())

# convert the lists of images and labels to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Use the flow method to apply augmentation
augmented_train_dataset = data_augmentation.flow(
    train_images[0], train_labels[0], batch_size=32
)


model = None
model = build_cnn_model(input_shape, num_classes)

# use the best combination that we found earlier
optimizer = copy.deepcopy(optimizers[best_combination[0]])

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the augmented_train_dataset dataset
# use the best combination that we found earlier
history = model.fit(augmented_train_dataset, epochs=10, validation_data=validation_dataset, batch_size=best_combination[1])

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''


🌟 Exercise 5: Performance Evaluation and Analysis
1. Objective: Thoroughly evaluate the model’s performance.
2. Task:

Plot the training and validation accuracy and loss curves. Analyze these plots for signs of overfitting or underfitting.
Calculate other relevant metrics like precision, recall, F1-score, and confusion matrix. How does the model perform on each flower class? Are there any classes that are particularly difficult to classify?
Visualize the model’s predictions on a set of test images. Identify any misclassifications and try to understand why they occurred.
'''

# Plot training and validation accuracy and loss curves
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')

plt.show()

# Generate predictions for the validation dataset
y_pred = model.predict(validation_dataset)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

# Extract true labels
y_true = []
for _, labels in validation_dataset:
    y_true.extend(labels.numpy())

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Classification report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()





