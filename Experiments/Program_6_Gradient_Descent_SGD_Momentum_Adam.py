
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_model():
    """Create an identical model for a fair optimizer comparison."""
    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    # 1. Generate and prepare a two-class dataset
    X, y = make_moons(n_samples=1200, noise=0.22, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # 2. Define optimizer configurations
    optimizers = {
        "Batch Gradient Descent": keras.optimizers.SGD(learning_rate=0.05),
        "Stochastic Gradient Descent": keras.optimizers.SGD(learning_rate=0.01),
        "Momentum": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "Adam": keras.optimizers.Adam(learning_rate=0.01)
    }

    histories = {}
    evaluations = {}

    # 3. Train a fresh, identically structured network for each optimizer
    for name, optimizer in optimizers.items():
        print(f"\\nTraining with {name}...")
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        model = build_model()
        # Use a fresh optimizer instance to avoid reusing optimizer state.
        if name == "Batch Gradient Descent":
            opt = keras.optimizers.SGD(learning_rate=0.05)
            batch_size = len(X_train)  # full-batch update
        elif name == "Stochastic Gradient Descent":
            opt = keras.optimizers.SGD(learning_rate=0.01)
            batch_size = 1
        elif name == "Momentum":
            opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
            batch_size = 32
        else:
            opt = keras.optimizers.Adam(learning_rate=0.01)
            batch_size = 32

        model.compile(
            optimizer=opt,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=35,
            batch_size=batch_size,
            verbose=0
        )
        histories[name] = history.history
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        evaluations[name] = (loss, accuracy)
        print(f"Test loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")

    # 4. Plot training loss and validation accuracy
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, history in histories.items():
        axes[0].plot(history["loss"], label=name)
        axes[1].plot(history["val_accuracy"], label=name)

    axes[0].set_title("Training Loss by Optimizer")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[1].set_title("Validation Accuracy by Optimizer")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # 5. Print final comparison table
    print("\\n--- FINAL OPTIMIZER COMPARISON ---")
    print(f"{'Optimizer':<30}{'Test Loss':<15}{'Test Accuracy':<15}")
    print("-" * 60)
    for name, (loss, accuracy) in evaluations.items():
        print(f"{name:<30}{loss:<15.4f}{accuracy:<15.4f}")

if __name__ == "__main__":
    main()
