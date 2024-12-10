from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import json
import os
from models import create_model_relu, create_model_leaky_relu, create_model_elu

# Function to load and preprocess the MNIST data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels

# Function to train and evaluate a model, and save its history
def train_and_evaluate_model(model, model_name):
    train_images, train_labels, test_images, test_labels = load_data()

    # Train the model and store the history
    history = model.fit(
        train_images,
        train_labels,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
    )

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    model.save(f'Models/{model_name}.h5')
    print(f"Model saved as Models/{model_name}.h5")

    # Save the history as a JSON file
    history_file = f'Histories/{model_name}_history.json'
    os.makedirs("Histories", exist_ok=True)  # Ensure the directory exists
    with open(history_file, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved as {history_file}")

    return model, history

# Main function to train and evaluate all models
def main():
    # Model 1: ReLU
    print("Training Model with ReLU activation")
    model_relu = create_model_relu()
    train_and_evaluate_model(model_relu, "model_relu")

    # Model 2: Leaky ReLU
    print("Training Model with Leaky ReLU activation")
    model_leaky_relu = create_model_leaky_relu()
    train_and_evaluate_model(model_leaky_relu, "model_leaky_relu")

    # Model 3: ELU
    print("Training Model with ELU activation")
    model_elu = create_model_elu()
    train_and_evaluate_model(model_elu, "model_elu")

if __name__ == '__main__':
    main()
