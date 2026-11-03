
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson

def main():
    # 1. Normal distribution
    x_normal = np.linspace(-4, 4, 500)
    normal_pdf = norm.pdf(x_normal, loc=0, scale=1)

    # 2. Binomial distribution: 10 trials, success probability 0.4
    x_binomial = np.arange(0, 11)
    binomial_pmf = binom.pmf(x_binomial, n=10, p=0.4)

    # 3. Poisson distribution: average 4 events
    x_poisson = np.arange(0, 15)
    poisson_pmf = poisson.pmf(x_poisson, mu=4)

    print("\\n--- DISTRIBUTION SUMMARY ---")
    print("Normal distribution: mean=0, standard deviation=1")
    print("P(-1 <= X <= 1) =", norm.cdf(1) - norm.cdf(-1))
    print("Binomial: n=10, p=0.4; mean =", binom.mean(10, 0.4))
    print("Poisson: lambda=4; mean =", poisson.mean(4))

    # 4. Bayes' theorem
    # P(D): prevalence; P(+|D): sensitivity; P(-|not D): specificity
    prevalence = 0.01
    sensitivity = 0.95
    specificity = 0.90
    p_positive = (sensitivity * prevalence +
                  (1 - specificity) * (1 - prevalence))
    p_disease_given_positive = sensitivity * prevalence / p_positive

    print("\\n--- BAYES' THEOREM: ILLUSTRATIVE TEST ---")
    print(f"P(Disease) = {prevalence:.3f}")
    print(f"P(Positive | Disease) = {sensitivity:.3f}")
    print(f"P(Negative | No Disease) = {specificity:.3f}")
    print(f"P(Positive) = {p_positive:.4f}")
    print("P(Disease | Positive) = "
          f"{p_disease_given_positive:.4f} ({p_disease_given_positive*100:.2f}%)")

    # 5. Plot probability distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(x_normal, normal_pdf)
    axes[0].set_title("Standard Normal PDF")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Probability density")

    axes[1].bar(x_binomial, binomial_pmf)
    axes[1].set_title("Binomial PMF (n=10, p=0.4)")
    axes[1].set_xlabel("Number of successes")
    axes[1].set_ylabel("Probability")

    axes[2].bar(x_poisson, poisson_pmf)
    axes[2].set_title("Poisson PMF (λ=4)")
    axes[2].set_xlabel("Number of events")
    axes[2].set_ylabel("Probability")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
