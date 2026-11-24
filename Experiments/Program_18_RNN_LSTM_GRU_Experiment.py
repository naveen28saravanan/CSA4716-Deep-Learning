import numpy as np
from tensorflow import keras
s=np.sin(np.arange(400)*.05).astype("float32"); w=18
X=np.array([s[i:i+w] for i in range(len(s)-w)])[...,None]; y=np.array([s[i+w] for i in range(len(s)-w)])
n=int(.8*len(X))
for name,cell in [("RNN",keras.layers.SimpleRNN),("LSTM",keras.layers.LSTM),("GRU",keras.layers.GRU)]:
 m=keras.Sequential([keras.layers.Input((w,1)),cell(16),keras.layers.Dense(1)])
 m.compile(optimizer=keras.optimizers.Adam(.005),loss="mse"); m.fit(X[:n],y[:n],epochs=7,verbose=0)
 print(name,"test MSE:",m.evaluate(X[n:],y[n:],verbose=0))
