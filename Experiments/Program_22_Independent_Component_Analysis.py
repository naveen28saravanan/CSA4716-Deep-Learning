import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
t=np.linspace(0,8,2000); S=np.c_[np.sin(2*t),np.sign(np.sin(3*t)),np.sin(5*t)]; S=(S-S.mean(0))/S.std(0)
A=np.array([[1,1,1],[.5,2,1],[1.5,1,2]]); X=S@A.T+.05*np.random.default_rng(1).normal(size=S.shape)
ica=FastICA(n_components=3,random_state=0,max_iter=1000); R=ica.fit_transform(X); print("Estimated mixing matrix:\n",ica.mixing_)
fig,ax=plt.subplots(3,1)
for i in range(3): ax[i].plot(R[:,i]); ax[i].set_ylabel(f"IC{i+1}")
plt.show()
