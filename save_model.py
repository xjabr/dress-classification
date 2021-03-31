from prepare_model import create_model, fit_model
from prepare_dataset import get_data_set

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels, test_images, test_labels, class_names) = get_data_set()

model = create_model()
fit_model(model, train_images, train_labels, 5)

# save models
model.summary()
model.evaluate(test_images, test_labels, verbose=2)

model.save('./model')