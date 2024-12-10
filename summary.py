import matplotlib.pyplot as plt
import json  # <-- Ensure json is imported
import h5py

def plot_training_vs_testing(model_name, history, metric="accuracy"):
    # Create a separate plot for each model
    plt.figure(figsize=(12, 8))

    colors = {
        "model_relu": "blue",
        "model_leaky_relu": "green",
        "model_elu": "orange",
        "model_sparsemax": "red",
    }
    
    color = colors.get(model_name, "black")
    
    # Plot training and validation
    plt.plot(history['train'][metric], label=f'{model_name} Train', color=color, linestyle='-')
    plt.plot(history['test'][metric], label=f'{model_name} Test', color=color, linestyle='--')

    # Set the title and labels
    plt.title(f"{model_name} - Training vs Testing {metric.capitalize()}", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot in the 'Graphs' folder
    plt.savefig(f"Graphs/{model_name}_{metric}_comparison.png")
    plt.close()  # Close the plot to avoid memory overflow during multiple saves

def load_histories():
    history_files = {
        "model_relu": "Histories/model_relu_history.json",
        "model_leaky_relu": "Histories/model_leaky_relu_history.json",
        "model_elu": "Histories/model_elu_history.json",
        "model_sparsemax": "Histories/model_sparsemax_history.json",
    }

    histories = {}
    for model_name, filepath in history_files.items():
        try:
            with open(filepath, "r") as f:
                history = json.load(f)  # Make sure this line uses the json module
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

def main():
    histories = load_histories()

    # Plot individual graphs for accuracy and loss
    for model_name, history in histories.items():
        plot_training_vs_testing(model_name, history, metric="accuracy")
        plot_training_vs_testing(model_name, history, metric="loss")

if __name__ == "__main__":
    main()
