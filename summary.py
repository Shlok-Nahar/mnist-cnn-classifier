import matplotlib.pyplot as plt
import json

def plot_training_vs_testing(model_name, history, metric="accuracy"):
    plt.figure(figsize=(12, 8))

    colors = {
        "model_relu": "blue",
        "model_leaky_relu": "green",
        "model_elu": "orange",
        "model_sparsemax": "red",
    }
    
    color = colors.get(model_name, "black")
    
    plt.plot(history['train'][metric], label=f'{model_name} Train', color=color, linestyle='-')
    plt.plot(history['test'][metric], label=f'{model_name} Test', color=color, linestyle='--')

    plt.title(f"{model_name} - Training vs Testing {metric.capitalize()}", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Graphs/{model_name}_{metric}_comparison.png")
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

def main():
    histories = load_histories()

    for model_name, history in histories.items():
        plot_training_vs_testing(model_name, history, metric="accuracy")
        plot_training_vs_testing(model_name, history, metric="loss")

if __name__ == "__main__":
    main()
