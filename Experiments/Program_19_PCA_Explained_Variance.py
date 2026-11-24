import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X,y=load_iris(return_X_y=True); X=StandardScaler().fit_transform(X); p=PCA().fit(X)
print("Eigenvalues:",p.explained_variance_); print("Variance ratios:",p.explained_variance_ratio_)
Z=p.transform(X); plt.scatter(Z[:,0],Z[:,1],c=y); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA Projection"); plt.show()
plt.plot(p.explained_variance_ratio_.cumsum(),marker="o"); plt.title("Cumulative Explained Variance"); plt.show()
