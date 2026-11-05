import numpy as np
import matplotlib.pyplot as plt
X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([0,0,0,1])
w=np.zeros(2); b=0.; lr=.1
for _ in range(20):
 for x,t in zip(X,y):
  p=int(x@w+b>=0); e=t-p; w+=lr*e*x; b+=lr*e
pred=(X@w+b>=0).astype(int)
print("Weights:",w,"Bias:",b,"Predictions:",pred,"Expected:",y)
plt.scatter(X[:,0],X[:,1],c=pred,s=150); plt.title("Perceptron AND Gate"); plt.grid(); plt.show()
