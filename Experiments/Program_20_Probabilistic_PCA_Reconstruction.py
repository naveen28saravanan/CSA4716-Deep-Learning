import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
d=load_digits(); s=StandardScaler(); X=s.fit_transform(d.data); p=PCA(n_components=20)
Z=p.fit_transform(X); R=s.inverse_transform(p.inverse_transform(Z))
print("Explained variance:",p.explained_variance_ratio_.sum()); print("Reconstruction MSE:",np.mean((d.data-R)**2))
fig,ax=plt.subplots(2,5)
for i in range(5):
 ax[0,i].imshow(d.images[i],cmap="gray"); ax[0,i].axis("off")
 ax[1,i].imshow(R[i].reshape(8,8),cmap="gray"); ax[1,i].axis("off")
plt.show()
