import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# Defining dataset paths
MNIST_TRAIN_CSV = "data/MNIST/mnist_train.csv"
MNIST_TEST_CSV = "data/MNIST/mnist_test.csv"
MATH_EXPRESSION_IMAGES_DIR = "data/handwritten_symbols/"
# Save paths
PROCESSED_DIR = "data/preprocessed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)


# preprocessing MNIST dataset
def preprocessing_MNIST(csv_path, dataset_type="your type of dataset"):
    """
    :param csv_path:
    :param dataset_type:
    load up csv file
    reshape the images to a 2d array (28x28) for the cnn model
    normalizing the pixel value of the image for the cnn model
    save the preprocessed data to the PROCESSED_DIR path
    """
    print(f'Loading {dataset_type} MNIST CSV dataset')

    df = pd.read_csv(csv_path)  # loading the csv file
    labels = df.iloc[:, 0].values  # Getting the digit label from the first column
    images = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)  # Reshaping the images from (784,) into 2D arrays (28x28)

    images = images / 255.0  # normalize the pixels values to [0,1]

    # Saving the preprocessed data
    np.save(os.path.join(PROCESSED_DIR, f"mnist_{dataset_type}.npy"), images)
    np.save(os.path.join(PROCESSED_DIR, f"mnist_{dataset_type}_labels.npy"), labels)

    print(f"MNIST {dataset_type} processed: {images.shape[0]} images saved.")


# preprocessing Handwritten dataset
def preprocessing_handwritten_expressions():
    """
    load up handwritten images
    loading the images in grayscale
    resizing the images to 64x64 for the cnn model
    normalize the pixel value
    converting the list to a NumPY array
    save the preprocessed data to the PROCESSED_DIR path
    """
    print(f'Processing Handwritten Math Expressions')

    image_paths = list(Path(MATH_EXPRESSION_IMAGES_DIR).glob("**/*.jpg")) + list(
        Path(MATH_EXPRESSION_IMAGES_DIR).glob("**/*.png"))
    labels = []
    images = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # load image in grayscale
        if img is None:
            print(f"Could not load {img_path}, so skipping img")
            continue
        img = cv2.resize(img, (64, 64))  # resizing the images to (64x64)
        img = img / 255.0  # normalizing the images for cnn model
        images.append(img)

        label = img_path.parent.name
        labels.append(label)

    images = np.array(images).reshape(-1, 64, 64, 1)  # converting the list to a NumPY array

    # Saving the preprocessed data
    np.save(os.path.join(PROCESSED_DIR, f"handwritten_expressions.npy"), images)
    np.save(os.path.join(PROCESSED_DIR, f"handwritten_expressions_labels.npy"), labels)
    print(f"Preprocessed {len(images)} handwritten expressions images into {len(labels)} successfully")


# run preprocessing
preprocessing_MNIST(MNIST_TRAIN_CSV, "train")
preprocessing_MNIST(MNIST_TEST_CSV, "test")
preprocessing_handwritten_expressions()

print("The preprocess step is complete")
