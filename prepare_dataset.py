import tensorflow as tf
from tensorflow import keras


def get_data_set():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = [ 'Maglietta / TOP', 'Pantaloni', 'Maglione', 'Vestito', 'Cappotto', 'Sandali', 'Camicia', 'Sneaker', 'Borsa', 'Stivali' ]

    # normalize values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels, test_images, test_labels, class_names)