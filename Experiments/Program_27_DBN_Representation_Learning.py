import matplotlib.pyplot as plt
from tensorflow import keras
(x,_),_=keras.datasets.mnist.load_data(); x=x[:8000].astype("float32").reshape(-1,784)/255
i=keras.Input((784,)); z=keras.layers.Dense(128,activation="relu")(i); z=keras.layers.Dense(32,activation="relu",name="latent")(z)
d=keras.layers.Dense(128,activation="relu")(z); o=keras.layers.Dense(784,activation="sigmoid")(d)
a=keras.Model(i,o); a.compile(optimizer="adam",loss="binary_crossentropy"); a.fit(x,x,epochs=4,batch_size=128,verbose=0)
enc=keras.Model(i,a.get_layer("latent").output); print("Representation shape:",enc.predict(x[:100],verbose=0).shape)
r=a.predict(x[:5],verbose=0); fig,ax=plt.subplots(2,5)
for j in range(5): ax[0,j].imshow(x[j].reshape(28,28),cmap="gray"); ax[1,j].imshow(r[j].reshape(28,28),cmap="gray"); ax[0,j].axis("off"); ax[1,j].axis("off")
plt.show()
print("Stacked autoencoder used as an educational latent-representation demonstration.")
