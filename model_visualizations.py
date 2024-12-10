import matplotlib.pyplot as plt
import tensorflow as tf
import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import label_binarize

def plot_training_vs_testing(histories, save_dir="Graphs"):
    # Colors for consistent distinction
    colors = {
        "model_relu": "blue",
        "model_leaky_relu": "green",
        "model_elu": "red"
    }

    # Create four separate plots
    metrics = ["accuracy", "loss"]
    datasets = ["train", "test"]

    for metric in metrics:
        for dataset in datasets:
            plt.figure(figsize=(12, 8))
            for model_name, history in histories.items():
                color = colors.get(model_name, "black")
                plt.plot(
                    history[dataset][metric], 
                    label=f'{model_name} {dataset.capitalize()}', 
                    color=color
                )

            plt.title(f"{dataset.capitalize()} {metric.capitalize()} for All Models", fontsize=16)
            plt.xlabel("Epochs", fontsize=14)
            plt.ylabel(metric.capitalize(), fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            save_path = f"{save_dir}/{dataset}_{metric}.png"
            plt.savefig(save_path)
            plt.close()

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
                history = json.load(f)
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

    test_labels_bin = label_binarize(test_labels.argmax(axis=1), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label='Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'ROC/{model_name}_roc_curve.png')
    plt.close()

def plot_precision_recall_curve(model, test_images, test_labels, model_name):
    y_pred = model.predict(test_images)

    test_labels_bin = label_binarize(test_labels.argmax(axis=1), classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(test_labels_bin[:, i], y_pred[:, i])

    plt.figure()
    for i in range(10):
        plt.plot(recall[i], precision[i], label='Class {0} (AP = {1:0.2f})'.format(i, average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="best")
    plt.savefig(f'Precision-Recall/{model_name}_precision_recall_curve.png')
    plt.close()

def plot_feature_maps(model, test_images, model_name):
    layer_outputs = [layer.output for layer in model.layers[:4]]
    feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    feature_maps = feature_map_model.predict(test_images[:1])

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
