import os

import matplotlib.pyplot as plt

from app.config import LOGS_DIR


def plot_training_history(history):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Evolution de la perte")
    plt.xlabel("Epoque")
    plt.ylabel("Perte")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history["val_auc"], label="Validation AUC")
    plt.axhline(y=0.5, color="r", linestyle="-", alpha=0.3, label="Random")
    plt.title("Evolution du ROC AUC")
    plt.xlabel("Epoque")
    plt.ylabel("AUC")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history["lr"], label="Learning Rate")
    plt.title("Evolution du taux d'apprentissage")
    plt.xlabel("Epoque")
    plt.ylabel("Taux d'apprentissage")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "enhanced_training_history.png"))
    plt.close()
