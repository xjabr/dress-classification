from prepare_dataset import get_data_set
from prepare_model import create_model, fit_model, random_predict, predictions

def main():
    (train_images, train_labels, test_images, test_labels, class_names) = get_data_set()

    model = create_model()
    fit_model(model, train_images, train_labels, epochs=5)
    # random_predict(model, test_images, test_labels, class_names)
    predictions(model, test_images, test_labels, class_names)

if __name__ == '__main__':
    main()