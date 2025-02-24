import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess_dataset import X_train, X_test, y_train, y_test, label_encoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Define CNN model
def create_cnn_model():
    """
    :return: model
    """
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Train CNN
cnn_model = create_cnn_model()
cnn_history = cnn_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save model
cnn_model.save('cnn_model.h5')

print("Model has been trained and saved to cnn_model.h5")
print("Now evaluating the cnn model")

# loading model for evaluation
loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
print(f'CNN Test accuracy: {cnn_accuracy * 100:.2f}%')


# Plot accuracy comparison
plt.figure(figsize=(12, 5))
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.title('CNN Accuracy training vs validation')
plt.legend()
plt.show()

