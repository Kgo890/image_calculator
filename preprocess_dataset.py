import os
import cv2
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
DATA_DIR = "data/dataset"
DATA_CSV = "data.csv"


# Function to extract images and labels
def extract_images(datadir):
    images_data = []
    images_label = []
    for folder in os.listdir(datadir):
        path = os.path.join(datadir, folder)
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))
            images_data.append(img)
            images_label.append(folder)
    combined = list(zip(images_data, images_label))
    random.shuffle(combined)
    images_data, images_label = zip(*combined)
    return np.array(images_data), np.array(images_label)


# Load dataset
images_data, images_label = extract_images(DATA_DIR)
X = images_data / 255.0

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(images_label)

# Save dataset
df_X = pd.DataFrame(X.reshape(X.shape[0], -1))
df_X['label'] = y
print("Saving extracted dataset to data.csv")
df_X.to_csv(DATA_CSV, index=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
