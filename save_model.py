from classification_dress import create_model, fit_model, predict

import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [ 'Maglietta / TOP', 'Pantaloni', 'Maglione', 'Vestito', 'Cappotto', 'Sandali', 'Camicia', 'Sneaker', 'Borsa', 'Stivali' ]

# normalize values
train_images = train_images / 255.0
test_images = test_images / 255.0

model = create_model()
fit_model(model, train_images, train_labels, 5)

# save models
model.summary()

model.save('./model')