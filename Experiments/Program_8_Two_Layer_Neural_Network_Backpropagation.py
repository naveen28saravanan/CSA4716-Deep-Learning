import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1); X=np.array([[0,0],[0,1],[1,0],[1,1]],float); y=np.array([[0],[1],[1],[0]],float)
sig=lambda z:1/(1+np.exp(-z)); W1=np.random.randn(2,4); W2=np.random.randn(4,1); b1=np.zeros((1,4)); b2=np.zeros((1,1)); losses=[]
for _ in range(8000):
 h=sig(X@W1+b1); o=sig(h@W2+b2); losses.append(np.mean((o-y)**2))
 d2=2*(o-y)*o*(1-o)/len(X); gW2=h.T@d2; gb2=d2.sum(0,keepdims=True)
 d1=(d2@W2.T)*h*(1-h); gW1=X.T@d1; gb1=d1.sum(0,keepdims=True)
 W2-=.8*gW2; b2-=.8*gb2; W1-=.8*gW1; b1-=.8*gb1
print("Predictions:\n",(o>=.5).astype(int),"\nExpected:\n",y.astype(int))
plt.plot(losses); plt.title("Backpropagation Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.show()
