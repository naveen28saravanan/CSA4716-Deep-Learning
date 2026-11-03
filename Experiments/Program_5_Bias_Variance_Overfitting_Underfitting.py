
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main():
    np.random.seed(7)

    # 1. Generate a nonlinear dataset with random noise
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = 0.5 * X[:, 0]**3 - 2 * X[:, 0]**2 + X[:, 0] + 3
    y = y + np.random.normal(0, 3.0, size=len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # 2. Compare models with increasing polynomial degree
    degrees = [1, 3, 15]
    results = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    x_curve = np.linspace(-3, 3, 300).reshape(-1, 1)

    for ax, degree in zip(axes, degrees):
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression()
        )
        model.fit(X_train, y_train)

        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        train_mse = mean_squared_error(y_train, train_prediction)
        test_mse = mean_squared_error(y_test, test_prediction)
        results.append((degree, train_mse, test_mse))

        curve_prediction = model.predict(x_curve)
        ax.scatter(X_train, y_train, s=18, alpha=0.55, label="Training data")
        ax.scatter(X_test, y_test, s=22, alpha=0.7, label="Test data")
        ax.plot(x_curve, curve_prediction, linewidth=2, label="Model")
        ax.set_title(f"Polynomial Degree {degree}")
        ax.set_xlabel("Input feature")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Target")
    axes[-1].legend()
    plt.suptitle("Underfitting, Suitable Fit and Overfitting")
    plt.tight_layout()
    plt.show()

    # 3. Display quantitative comparison
    print("\\n--- MODEL PERFORMANCE ---")
    print(f"{'Degree':<10}{'Training MSE':<18}{'Testing MSE':<18}")
    print("-" * 46)
    for degree, train_mse, test_mse in results:
        print(f"{degree:<10}{train_mse:<18.4f}{test_mse:<18.4f}")

    print("\\nInterpretation:")
    print("- Degree 1 is intentionally simple and may underfit the nonlinear pattern.")
    print("- Degree 3 can represent the underlying cubic relationship more suitably.")
    print("- Degree 15 is highly flexible and may fit noise, indicating overfitting.")
    print("- A large gap between training and testing error is a warning sign.")

if __name__ == "__main__":
    main()
