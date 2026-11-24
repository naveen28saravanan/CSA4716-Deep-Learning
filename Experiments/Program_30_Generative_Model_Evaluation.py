import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error,mean_absolute_error
d=load_digits(); X=d.data/16.; rows=[]; rec=[]
for n in [8,24]:
 p=PCA(n_components=n,random_state=1); R=p.inverse_transform(p.fit_transform(X)); rec.append(R)
 rows.append({"Model":f"PCA-{n}","MSE":mean_squared_error(X,R),"MAE":mean_absolute_error(X,R),"Explained variance":p.explained_variance_ratio_.sum()})
table=pd.DataFrame(rows); print(table.to_string(index=False))
fig,ax=plt.subplots(1,3)
for a1,img,title in zip(ax,[X[0],rec[0][0],rec[1][0]],["Original","PCA-8","PCA-24"]): a1.imshow(img.reshape(8,8),cmap="gray"); a1.set_title(title); a1.axis("off")
plt.show(); table.set_index("Model")[["MSE","MAE"]].plot(kind="bar"); plt.title("Reconstruction Metrics"); plt.show()
