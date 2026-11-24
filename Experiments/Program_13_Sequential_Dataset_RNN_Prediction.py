import numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
s=np.sin(np.arange(500)*.04)+.25*np.sin(np.arange(500)*.13); w=20
X=np.array([s[i:i+w] for i in range(len(s)-w)])[...,None]; y=np.array([s[i+w] for i in range(len(s)-w)])
n=int(.8*len(X)); m=keras.Sequential([keras.layers.Input((w,1)),keras.layers.SimpleRNN(32),keras.layers.Dense(1)])
m.compile(optimizer="adam",loss="mse"); m.fit(X[:n],y[:n],epochs=15,verbose=0)
p=m.predict(X[n:],verbose=0).ravel(); print("Test MSE:",np.mean((p-y[n:])**2))
plt.plot(y[n:],label="Actual"); plt.plot(p,label="Predicted"); plt.legend(); plt.title("RNN Prediction"); plt.show()
