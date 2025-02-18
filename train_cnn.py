import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# load the preprocess data
PROCESSED_DIR = "data/preprocessed"
# for MNIST
print("Loading the preprocessed MNIST data")
X_digits = np.load(os.path.join(PROCESSED_DIR, "mnist_train.npy"))
y_digits = np.load(os.path.join(PROCESSED_DIR, "mnist_train_labels.npy"))

print("Loading the preprocessed Handwritten data")
X_symbols = np.load(os.path.join(PROCESSED_DIR, "handwritten_expressions.npy"))
y_symbols = np.load(os.path.join(PROCESSED_DIR, "handwritten_expressions_labels.npy"), allow_pickle=True)

# convert labels to categorical (one hot encoding)
y_digits = to_categorical(y_digits, num_classes=10)  # digits (0-9)

# encode math labels
unique_labels = sorted(set(y_symbols))  # Get unique symbol classes
label_map = {label: i for i, label in enumerate(unique_labels)}  # Map symbols to numbers
y_symbols = np.array([label_map[label] for label in y_symbols])  # Convert labels
y_symbols = to_categorical(y_symbols, num_classes=len(unique_labels))

# splitting into training and testing
# MNIST dataset
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.2,
                                                                                random_state=42)
# Handwritten dataset
X_train_symbols, X_test_symbols, y_train_symbols, y_test_symbol = train_test_split(X_symbols, y_symbols, test_size=0.2,
                                                                                   random_state=42)
# splitting for validation data set from the training dataset
X_train_digits, X_valid_digits, y_train_digits, y_valid_digits = train_test_split(X_train_digits, y_train_digits,
                                                                                  test_size=0.1, random_state=42)
X_train_symbols, X_valid_symbols, y_train_symbols, y_valid_symbols = train_test_split(X_train_symbols, y_train_symbols,
                                                                                      test_size=0.1, random_state=42)


# create CNN model and compile model
def cnn_model(input_shape, num_classes):
    """
    :param input_shape:
    :param num_classes:
    random seed to get the same results
    input_layer: 32 filters with a filter size of 3x3 and use relu
    first max pooling layer: reducing the image size
    second_layer: 64 filters with filter size of 3x3 and use relu
    second max pooling layer: reduce the image size
    flatten layer: turn 2d feature map into 1d vector
    first dense layer: use 128 neurons for a balance relationship, use relu
    second dropout layer: prevent overfitting
    output layer: num_class for the final classification, softmax because multi classes

    compile: categorical_crossentrophy, adam as an optimizer, metric accuracy for classification problem
    :return: model
    """
    tf.random.set_seed(42)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # mutil-class classification problem

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# creating the models
digit_model = cnn_model((28, 28, 1), 10)  # model for digits
symbols_model = cnn_model((64, 64, 1), len(unique_labels))  # model for symbols

# check the summary of the models
digit_model.summary()
symbols_model.summary()

# training models
"""
batch size = 32 for balance speed and accuracy
"""
print("Now training the MNIST dataset")
digit_model.fit(X_train_digits, y_train_digits, epochs=10, validation_data=(X_valid_digits, y_valid_digits),
                batch_size=32)
print("Now training the Handwritten dataset")
symbols_model.fit(X_train_symbols, y_train_symbols, epochs=10, validation_data=(X_valid_symbols, y_valid_symbols),
                  batch_size=32)
# save model
digit_model.save(os.path.join(PROCESSED_DIR, "digit_cnn_model.h5"))
symbols_model.save(os.path.join(PROCESSED_DIR, "symbol_cnn_model.h5"))
