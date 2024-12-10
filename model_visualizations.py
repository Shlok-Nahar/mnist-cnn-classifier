import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer

def plot_training_vs_testing(histories, metric="accuracy", save_dir="Graphs/Accuracy"):
    plt.figure(figsize=(12, 8))

    # Colors for each model
    colors = {
        "model_relu": "blue",
        "model_leaky_relu": "green",
        "model_elu": "red"
    }
    
    for model_name, history in histories.items():
        color = colors.get(model_name, "black")
        
        # Plot training and validation
        plt.plot(history['train'][metric], label=f'{model_name} Train', color=color, linestyle='-')
        plt.plot(history['test'][metric], label=f'{model_name} Test', color=color, linestyle='--')

    plt.title(f"Training vs Testing {metric.capitalize()} Comparison", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot in the appropriate folder
    plt.savefig(f"{save_dir}/{metric}_comparison.png")
    plt.close()  # Close to avoid memory issues

def load_histories():
    history_files = {
        "model_relu": "Histories/model_relu_history.json",
        "model_leaky_relu": "Histories/model_leaky_relu_history.json",
        "model_elu": "Histories/model_elu_history.json"
    }

    histories = {}
    for model_name, filepath in history_files.items():
        try:
            with open(filepath, "r") as f:
                history = json.load(f)  # Load the history
                histories[model_name] = {
                    "train": {
                        "accuracy": history.get("accuracy", []),
                        "loss": history.get("loss", []),
                    },
                    "test": {
                        "accuracy": history.get("val_accuracy", []),
                        "loss": history.get("val_loss", []),
                    },
                }
        except Exception as e:
            print(f"Could not load history for {model_name}: {e}")

    return histories

def plot_confusion_matrix(model, test_images, test_labels, model_name):
    y_pred = model.predict(test_images)
    cm = confusion_matrix(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"Confusion Matrix/{model_name}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(model, test_images, test_labels, model_name):
    y_pred = model.predict(test_images)
    fpr, tpr, _ = roc_curve(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f"{model_name} Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"ROC/{model_name}_roc_curve.png")
    plt.close()

def plot_precision_recall_curve(model, test_images, test_labels, model_name):
    y_pred = model.predict(test_images)
    precision, recall, _ = precision_recall_curve(test_labels.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', lw=2)
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"Precision-Recall/{model_name}_precision_recall_curve.png")
    plt.close()

def plot_feature_maps(model, test_images, model_name):
    layer_outputs = [layer.output for layer in model.layers[:4]]  # Taking first few layers
    feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    feature_maps = feature_map_model.predict(test_images[:1])  # Get feature maps for the first test image
    
    for layer_idx, feature_map in enumerate(feature_maps):
        num_filters = feature_map.shape[-1]
        size = feature_map.shape[1]
        
        display_grid = np.zeros((size, size * num_filters))
        for filter_idx in range(num_filters):
            x = feature_map[0, :, :, filter_idx]
            x -= x.mean()
            x /= x.std()
            x *= 255
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, filter_idx * size : (filter_idx + 1) * size] = x
        
        plt.figure(figsize=(10, 10))
        plt.title(f"{model_name} - Feature Maps of Layer {layer_idx+1}")
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.savefig(f"Heatmap/{model_name}_layer{layer_idx+1}_feature_map.png")
        plt.close()

def plot_model_architecture(model, model_name):
    plot_model(model, to_file=f"Model Architecture/{model_name}_architecture.png", show_shapes=True, show_layer_names=True)
