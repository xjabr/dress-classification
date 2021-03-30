import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import random

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def fit_model(model, images, labels, epochs):
    model.fit(images, labels, epochs=epochs)

def predict(model, images, labels, class_names):
    """
    Generate a random index from 0 to N-1 of images' array and prelev
    image from the index generated and start the predict, when it is done
    check if the prediction label is equals to the real label
    """
    n_tests = len(images)
    random_index = random.randint(0, n_tests-1)

    print('Random Index ', random_index)

    img = images[random_index]
    img = (np.expand_dims(img, 0))

    prediction = model.predict(img)
    true_label = labels[random_index]
    predicted_label = np.argmax(prediction)

    print('True: {}\nPredicted: {}'.format(class_names[true_label], class_names[predicted_label]))

    # show image
    plt.figure(figsize=(10, 10))
    plt.imshow(images[random_index])

    plt.show()