import os
from tensorflow.keras.models import load_model
from model_visualizations import load_histories, plot_training_vs_testing, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_feature_maps, plot_model_architecture
from main import load_data
import tensorflow as tf

directories = [
    "Graphs/Accuracy", "Graphs/Loss", 
    "Confusion Matrix", "ROC", 
    "Precision-Recall", "Heatmap", 
    "Model Architecture"
]
for directory in directories:
    os.makedirs(directory, exist_ok=True)

def main():
    histories = load_histories()

    plot_training_vs_testing(histories, save_dir="Graphs")

    models = {
        "model_relu": load_model("Models/model_relu.h5"),
        "model_leaky_relu": load_model("Models/model_leaky_relu.h5"),
        "model_elu": load_model("Models/model_elu.h5")
    }

    _, _, test_images, test_labels = load_data()

    for model_name, model in models.items():
        # Confusion Matrix
        plot_confusion_matrix(model, test_images, test_labels, model_name)
        
        # ROC Curve
        plot_roc_curve(model, test_images, test_labels, model_name)
        
        # Precision-Recall Curve
        plot_precision_recall_curve(model, test_images, test_labels, model_name)
        
        # Feature Maps
        plot_feature_maps(model, test_images, model_name)

if __name__ == "__main__":
    main()
