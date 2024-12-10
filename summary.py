import tensorflow as tf
from tensorflow.keras.models import load_model

def print_model_summary(model_path):
    model = load_model(model_path)

    print(f"Model Summary for {model_path}:")
    model.summary()
    print("\n")

def main():
    model_paths = [
        'Models/model_relu.h5',
        'Models/model_leaky_relu.h5',
        'Models/model_elu.h5',
        'Models/model_sparsemax.h5'
    ]

    for model_path in model_paths:
        print_model_summary(model_path)

if __name__ == '__main__':
    main()
