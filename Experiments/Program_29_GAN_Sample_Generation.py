import numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
(x,_),_=keras.datasets.mnist.load_data(); x=x[:8000].astype("float32").reshape(-1,784)/127.5-1
zdim=32
g=keras.Sequential([layers.Input((zdim,)),layers.Dense(128,activation="relu"),layers.Dense(784,activation="tanh")])
d=keras.Sequential([layers.Input((784,)),layers.Dense(128),layers.LeakyReLU(),layers.Dense(1,activation="sigmoid")])
d.compile(optimizer=keras.optimizers.Adam(.0002),loss="binary_crossentropy"); d.trainable=False
z=layers.Input((zdim,)); gan=keras.Model(z,d(g(z))); gan.compile(optimizer=keras.optimizers.Adam(.0002),loss="binary_crossentropy")
rng=np.random.default_rng(3)
for step in range(250):
 real=x[rng.integers(0,len(x),32)]; noise=rng.normal(size=(32,zdim)).astype("float32"); fake=g.predict(noise,verbose=0)
 d.trainable=True; d.train_on_batch(real,np.ones((32,1))); d.train_on_batch(fake,np.zeros((32,1))); d.trainable=False
 gan.train_on_batch(rng.normal(size=(32,zdim)).astype("float32"),np.ones((32,1)))
 if (step+1)%50==0: print("Step",step+1)
samples=g.predict(rng.normal(size=(16,zdim)).astype("float32"),verbose=0); fig,ax=plt.subplots(4,4)
for a1,img in zip(ax.flat,samples): a1.imshow((img.reshape(28,28)+1)/2,cmap="gray"); a1.axis("off")
plt.show()
