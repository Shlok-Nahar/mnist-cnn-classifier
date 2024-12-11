import json

def generate_comparison_table(histories):
    table_data = []

    for model_name, history in histories.items():
        train_accuracy = history['train']['accuracy'][-1] if history['train']['accuracy'] else None
        test_accuracy = history['test']['accuracy'][-1] if history['test']['accuracy'] else None
        train_loss = history['train']['loss'][-1] if history['train']['loss'] else None
        test_loss = history['test']['loss'][-1] if history['test']['loss'] else None

        table_data.append({
            "Model": model_name,
            "Accuracy Train": train_accuracy,
            "Accuracy Test": test_accuracy,
            "Loss Train": train_loss,
            "Loss Test": test_loss
        })

    # Print the table
    print("\nModel Performance Comparison")
    print("=" * 50)
    print(f"{'Model':<20}{'Metric':<10}{'Train':<10}{'Test':<10}")
    print("-" * 50)

    for row in table_data:
        print(f"{row['Model']:<20}{'Accuracy':<10}{row['Accuracy Train']:<10.4f}{row['Accuracy Test']:<10.4f}")
        print(f"{'':<20}{'Loss':<10}{row['Loss Train']:<10.4f}{row['Loss Test']:<10.4f}")

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
    generate_comparison_table(histories)

if __name__ == "__main__":
    main()
