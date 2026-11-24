import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
d=load_wine(); X=StandardScaler().fit_transform(d.data); f=FactorAnalysis(n_components=3,random_state=1)
scores=f.fit_transform(X); L=pd.DataFrame(f.components_.T,index=d.feature_names,columns=["Factor 1","Factor 2","Factor 3"])
print(L.round(3)); print("Latent score shape:",scores.shape)
plt.imshow(L,cmap="coolwarm",aspect="auto"); plt.colorbar(); plt.yticks(range(len(L)),L.index,fontsize=7); plt.xticks(range(3),L.columns); plt.show()
