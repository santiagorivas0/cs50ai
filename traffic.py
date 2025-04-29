import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Constants
NUM_CATEGORIES = 43
IMG_WIDTH = 30
IMG_HEIGHT = 30
EPOCHS = 10
TEST_SIZE = 0.4

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    Return tuple `(images, labels)`.
    `images` should be a list of all the images in the data set,
    each represented as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3.
    `labels` should be a list of integers representing the category for each image.
    """
    images = []
    labels = []
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(category)
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = keras.Sequential([
        # Convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Another convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and add dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def main():
    import sys
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Load data
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}")

if __name__ == "__main__":
    main()
