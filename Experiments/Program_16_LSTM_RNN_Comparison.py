import numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
s=np.sin(np.arange(500)*.04); w=20
X=np.array([s[i:i+w] for i in range(len(s)-w)])[...,None]; y=np.array([s[i+w] for i in range(len(s)-w)])
n=int(.8*len(X))
for name,cell in [("RNN",keras.layers.SimpleRNN),("LSTM",keras.layers.LSTM)]:
 m=keras.Sequential([keras.layers.Input((w,1)),cell(24),keras.layers.Dense(1)])
 m.compile(optimizer="adam",loss="mse"); h=m.fit(X[:n],y[:n],epochs=12,validation_split=.1,verbose=0)
 p=m.predict(X[n:],verbose=0).ravel(); print(name,"test MSE",np.mean((p-y[n:])**2))
 plt.plot(h.history["val_loss"],label=name)
plt.title("RNN vs LSTM Validation Loss"); plt.legend(); plt.show()
