import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import load_json


def plot_training_curves(history_path: Path, output_dir: Path) -> None:
    history = load_json(history_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()

    # F1 macro curve if available
    if "val_f1_macro" in history:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history["val_f1_macro"], label="Val F1 (macro)")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Validation F1 (macro)")
        plt.legend()

    output_path = output_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm_csv_path: Path, output_dir: Path) -> None:
    cm_df = pd.read_csv(cm_csv_path, index_col=0)
    labels = list(cm_df.index)
    cm = cm_df.values

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    output_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize training metrics and confusion matrix")
    parser.add_argument("--artifacts_dir", type=str, default="outputs")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    history_path = artifacts_dir / "training_history.json"
    cm_csv_path = artifacts_dir / "confusion_matrix.csv"

    if history_path.exists():
        plot_training_curves(history_path, artifacts_dir)
        print(f"Saved training curves to {artifacts_dir / 'training_curves.png'}")
    else:
        print("No training_history.json found. Skipping curves plot.")

    if cm_csv_path.exists():
        plot_confusion_matrix(cm_csv_path, artifacts_dir)
        print(f"Saved confusion matrix to {artifacts_dir / 'confusion_matrix.png'}")
    else:
        print("No confusion_matrix.csv found. Train first to generate artifacts.")


if __name__ == "__main__":
    main()


