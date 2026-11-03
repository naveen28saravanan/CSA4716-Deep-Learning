
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.set_printoptions(precision=3, suppress=True)

    # 1. Create vectors and matrices
    vector_a = np.array([2, 4, 6], dtype=float)
    vector_b = np.array([1, 3, 5], dtype=float)
    matrix_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    matrix_b = np.array([[6, 5, 4], [3, 2, 1]], dtype=float)

    print("\\n--- VECTOR OPERATIONS ---")
    print("Vector A:", vector_a)
    print("Vector B:", vector_b)
    print("Addition:", vector_a + vector_b)
    print("Subtraction:", vector_a - vector_b)
    print("Element-wise multiplication:", vector_a * vector_b)
    print("Dot product:", np.dot(vector_a, vector_b))
    print("Magnitude (L2 norm) of A:", np.linalg.norm(vector_a))
    print("Unit vector A:", vector_a / np.linalg.norm(vector_a))

    print("\\n--- MATRIX OPERATIONS ---")
    print("Matrix A:\\n", matrix_a)
    print("Matrix B:\\n", matrix_b)
    print("Addition:\\n", matrix_a + matrix_b)
    print("Element-wise multiplication:\\n", matrix_a * matrix_b)
    print("Transpose of A:\\n", matrix_a.T)
    print("Matrix multiplication A @ B.T:\\n", matrix_a @ matrix_b.T)

    # 2. Square matrix operations
    square_matrix = np.array([[4, 2], [1, 3]], dtype=float)
    print("\\nSquare matrix:\\n", square_matrix)
    print("Determinant:", np.linalg.det(square_matrix))
    print("Inverse:\\n", np.linalg.inv(square_matrix))
    print("Eigenvalues:", np.linalg.eigvals(square_matrix))

    # 3. A simple linear transformation: y = Wx + b
    weights = np.array([[0.2, 0.8], [0.5, -0.3]])
    inputs = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]])
    bias = np.array([0.1, 0.2])
    outputs = inputs @ weights.T + bias
    print("\\nInputs:\\n", inputs)
    print("Weights:\\n", weights)
    print("Bias:", bias)
    print("Linear layer output (X @ W.T + b):\\n", outputs)

    # 4. Visualize matrix values and transformed outputs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(matrix_a, cmap="viridis")
    axes[0].set_title("Matrix A")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    for (i, j), value in np.ndenumerate(matrix_a):
        axes[0].text(j, i, f"{value:g}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=axes[0])

    axes[1].plot(outputs[:, 0], marker="o", label="Output neuron 1")
    axes[1].plot(outputs[:, 1], marker="s", label="Output neuron 2")
    axes[1].set_title("Output of a Linear Layer")
    axes[1].set_xlabel("Input sample")
    axes[1].set_ylabel("Output value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
