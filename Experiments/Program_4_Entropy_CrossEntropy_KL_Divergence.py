
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def cross_entropy(p, q):
    """H(P,Q) = -sum p(x) log(q(x)); clips q to avoid log(0)."""
    q_safe = np.clip(q, 1e-12, 1.0)
    return -np.sum(p * np.log(q_safe))

def main():
    # P represents the target distribution; Q is a predicted distribution.
    p = np.array([0.50, 0.30, 0.20], dtype=float)
    q_good = np.array([0.45, 0.35, 0.20], dtype=float)
    q_poor = np.array([0.10, 0.20, 0.70], dtype=float)

    # Ensure valid probability distributions
    for name, distribution in [("P", p), ("Q good", q_good), ("Q poor", q_poor)]:
        if np.any(distribution < 0) or not np.isclose(distribution.sum(), 1):
            raise ValueError(f"{name} must contain non-negative probabilities summing to 1.")

    h_p = entropy(p)
    h_q_good = cross_entropy(p, q_good)
    h_q_poor = cross_entropy(p, q_poor)
    kl_good = entropy(p, q_good)
    kl_poor = entropy(p, q_poor)

    print("\\n--- PROBABILITY DISTRIBUTIONS ---")
    print("Target P:", p)
    print("Predicted Q (good):", q_good)
    print("Predicted Q (poor):", q_poor)

    print("\\n--- INFORMATION-THEORY METRICS (nats) ---")
    print(f"Entropy H(P):                    {h_p:.6f}")
    print(f"Cross-Entropy H(P, Q good):       {h_q_good:.6f}")
    print(f"Cross-Entropy H(P, Q poor):       {h_q_poor:.6f}")
    print(f"KL Divergence D(P || Q good):     {kl_good:.6f}")
    print(f"KL Divergence D(P || Q poor):     {kl_poor:.6f}")

    # Verify the identity: H(P,Q) = H(P) + D_KL(P||Q)
    print("\\nIdentity check for good Q:",
          np.isclose(h_q_good, h_p + kl_good))

    # Visual comparison of distributions and metric values
    labels = ["Class A", "Class B", "Class C"]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(x - width, p, width, label="Target P")
    axes[0].bar(x, q_good, width, label="Predicted Q (good)")
    axes[0].bar(x + width, q_poor, width, label="Predicted Q (poor)")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Probability")
    axes[0].set_title("Target vs Predicted Distributions")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)

    metric_names = ["Entropy P", "Cross-entropy\\ngood Q", "Cross-entropy\\npoor Q"]
    metric_values = [h_p, h_q_good, h_q_poor]
    axes[1].bar(metric_names, metric_values)
    axes[1].set_ylabel("Value (nats)")
    axes[1].set_title("Entropy and Cross-Entropy")
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
