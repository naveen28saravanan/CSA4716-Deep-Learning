
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    # 1. Generate correlated observations for two features
    mean = [5, 8]
    covariance_source = [[4, 3.2], [3.2, 5]]
    data = np.random.multivariate_normal(mean, covariance_source, size=300)
    df = pd.DataFrame(data, columns=["Feature_1", "Feature_2"])

    print("\\n--- DATASET PREVIEW ---")
    print(df.head())
    print("\\nDescriptive statistics:\\n", df.describe())

    # 2. Calculate covariance and correlation
    cov_matrix = np.cov(data, rowvar=False)
    corr_matrix = df.corr()
    print("\\nCovariance matrix:\\n", cov_matrix)
    print("\\nCorrelation matrix:\\n", corr_matrix)

    # 3. Eigenvalue and eigenvector calculation
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    print("\\nEigenvalues (descending):", eigenvalues)
    print("Eigenvectors (columns):\\n", eigenvectors)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    print("Explained variance ratio:", explained_variance_ratio)
    print("Total explained variance:", explained_variance_ratio.sum())

    # 4. Plot observations and eigenvector directions
    center = data.mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.45, s=18)
    axes[0].scatter(*center, color="red", marker="x", s=90, label="Mean")
    scale = 2.5
    for i in range(2):
        direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * scale
        axes[0].arrow(center[0], center[1], direction[0], direction[1],
                      width=0.025, head_width=0.18, length_includes_head=True,
                      label=f"Eigenvector {i+1}")
    axes[0].set_title("Data and Principal Directions")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(["PC1", "PC2"], explained_variance_ratio)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Proportion of variance")
    axes[1].set_title("Explained Variance Ratio")
    for i, value in enumerate(explained_variance_ratio):
        axes[1].text(i, value + 0.025, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
