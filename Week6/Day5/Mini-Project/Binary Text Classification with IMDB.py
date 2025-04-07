'''


Binary Text Classification with IMDB


Mini-Project: Binary Text Classification with IMDB Dataset (CNN)


What You'll learn
How to preprocess text data for neural networks.
How to build and train a simple feedforward neural network for binary classification.
How to evaluate the performance of a model using validation and test data.
How to visualize training and validation metrics to detect overfitting.


What you will create
A binary text classification model using the IMDB dataset to classify movie reviews as positive or negative.
A visualization of training and validation loss and accuracy to analyze model performance.


Dataset
The dataset used in this project is the IMDB Movie Reviews Dataset, which contains 50,000 reviews labeled as positive (1) or negative (0). The dataset is preprocessed, with each review encoded as a sequence of integers representing the most frequent 10,000 words in the dataset.
You can find it here : IMDB Dataset



'''
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

1. Preprocess the Data:

Load the IMDB dataset using Keras.
Convert the sequences of integers into binary matrices using one-hot encoding.
Split the data into training, validation, and test sets.

'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras import models, layers, losses, metrics, optimizers
import matplotlib.pyplot as plt


# Load the CSV file
data = pd.read_csv('IMDB Dataset.csv')

# the CSV has columns "review" for text and "sentiment" (negative-0 or positive-1)
reviews = data['review'].astype(str).tolist()  # text data is string type
# Map text labels to numeric values
data['sentiment'] = data['sentiment'].map({'negative': 0, 'positive': 1})
labels = data[('sentiment')].tolist()

# Split the data into training and testing sets
x_train_texts, x_test_texts, y_train, y_test = train_test_split(
    reviews, labels, test_size=0.2, random_state=42)

# Tokenize the text with a vocabulary limited to 10000 words
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train_texts)  # Fit only on training data

# Convert texts to sequences of integers
x_train_seq = tokenizer.texts_to_sequences(x_train_texts)
x_test_seq = tokenizer.texts_to_sequences(x_test_texts)

# Vectorize the sequences into one-hot encoded binary vectors
def vectorize_sequences(sequences, dimension=max_words):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for index in sequence:
            # Make sure the index is within the vocabulary range
            if index < dimension:
                results[i, index] = 1.
    return results

x_train = vectorize_sequences(x_train_seq, max_words)
x_test = vectorize_sequences(x_test_seq, max_words)

# Convert labels to numpy arrays with float type
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

2. Build the Model:

Create a feedforward neural network with:
Two hidden layers using ReLU activation.
An output layer with a sigmoid activation for binary classification.
Compile the model using the RMSprop optimizer, binary cross-entropy loss, and accuracy as the evaluation metric.

'''


# Build the feedforward neural network
model = models.Sequential()

# Add the first hidden layer
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))

# Add the second hidden layer
model.add(layers.Dense(64, activation='relu'))

# Add the output layer
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

3. Train the Model:

Train the model on the training data for 20 epochs with a batch size of 512.
Use the validation set to monitor performance during training.

'''

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=512,
    validation_split=0.2
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

4. Evaluate the Model:

Plot the training and validation loss and accuracy to detect overfitting.
Retrain the model with an optimal number of epochs to avoid overfitting.
Evaluate the final model on the test set to measure its performance.

'''

# Plot the training and validation loss and accuracy
loss = history.history['loss']
accuracy = history.history['accuracy']

epochs = range(1, len(loss) + 1)

# Plot loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# Retrain the model with an optimal number of epochs to avoid overfitting
optimal_epochs = 15  # I can see in the plot that there's no need to have more than 15 epochs
history = model.fit(
            x_train, y_train,
            epochs=optimal_epochs,
            batch_size=512,
            validation_split=0.2
)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

5. Analyze Results:

Compare the training and validation metrics to understand the model's behavior.
Report the final accuracy and loss on the test set.

'''

# Compare training and validation metrics

loss = history.history['loss']
accuracy = history.history['accuracy']

val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

optimal_epochs = range(1, optimal_epochs + 1)

# Plot validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(optimal_epochs, val_loss, 'ro-', label='Validation loss')
plt.plot(optimal_epochs, loss, 'bo-', label='Training loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(optimal_epochs, val_accuracy, 'ro-', label='Validation accuracy')
plt.plot(optimal_epochs, accuracy, 'bo-', label='Training accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Report final test accuracy and loss
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

Conclusion
In this mini-project, you built and trained a binary text classification model using the IMDB dataset. You learned how to preprocess text data, design a neural network, and evaluate its performance using validation and test sets. By visualizing the training and validation metrics, you also gained insight into the importance of avoiding overfitting. This project serves as a foundation for more advanced natural language processing tasks, such as sentiment analysis, text generation, and sequence modeling.


'''


