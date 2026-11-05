import matplotlib.pyplot as plt
from tensorflow import keras
(x,y),(xt,yt)=keras.datasets.fashion_mnist.load_data(); x=x[...,None]/255.; xt=xt[...,None]/255.
m=keras.Sequential([keras.layers.Input((28,28,1)),keras.layers.Conv2D(32,3,activation="relu"),keras.layers.MaxPooling2D(),keras.layers.Conv2D(64,3,activation="relu"),keras.layers.MaxPooling2D(),keras.layers.Flatten(),keras.layers.Dense(64,activation="relu"),keras.layers.Dense(10,activation="softmax")])
m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); m.fit(x,y,epochs=4,batch_size=128,validation_split=.1)
print("Test metrics:",m.evaluate(xt,yt,verbose=0))
p=m.predict(xt[:10],verbose=0).argmax(1); fig,ax=plt.subplots(2,5)
for a1,img,t,q in zip(ax.flat,xt[:10],yt[:10],p): a1.imshow(img.squeeze(),cmap="gray"); a1.set_title(f"T{t}/P{q}"); a1.axis("off")
plt.tight_layout(); plt.show()
