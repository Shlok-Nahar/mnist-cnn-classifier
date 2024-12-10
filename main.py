import tensorflow as tf
import sys
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from models import create_model_leaky_relu, create_model_elu, create_model_sparsemax

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels

def train_and_evaluate_model(model, model_name):
    train_images, train_labels, test_images, test_labels = load_data()

    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
    
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    model.save(f'Models/{model_name}.h5')
    print(f"Model saved as {model_name}.h5")
    
    return model

def main():
    # Model 1: Leaky ReLU
    print("Training Model with Leaky ReLU activation")
    model_leaky_relu = create_model_leaky_relu()
    train_and_evaluate_model(model_leaky_relu, "model_leaky_relu")

    # Model 2: ELU
    print("Training Model with ELU activation")
    model_elu = create_model_elu()
    train_and_evaluate_model(model_elu, "model_elu")

    # Model 3: SparseMax
    print("Training Model with SparseMax activation")
    model_sparsemax = create_model_sparsemax()
    train_and_evaluate_model(model_sparsemax, "model_sparsemax")

if __name__ == '__main__':
    main()
