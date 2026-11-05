import matplotlib.pyplot as plt
from tensorflow import keras
(x,y),(xt,yt)=keras.datasets.mnist.load_data(); x=x[:8000,...,None]/255.; y=y[:8000]; xt=xt[:1500,...,None]/255.; yt=yt[:1500]
scores=[]
for f,k,u in [(16,3,32),(32,3,64),(32,5,128)]:
 m=keras.Sequential([keras.layers.Input((28,28,1)),keras.layers.Conv2D(f,k,activation="relu"),keras.layers.MaxPooling2D(),keras.layers.Flatten(),keras.layers.Dense(u,activation="relu"),keras.layers.Dense(10,activation="softmax")])
 m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); m.fit(x,y,epochs=2,batch_size=128,verbose=0)
 loss,acc=m.evaluate(xt,yt,verbose=0); scores.append(acc); print((f,k,u),"accuracy",acc)
plt.bar(["16/3/32","32/3/64","32/5/128"],scores); plt.title("CNN Hyperparameter Comparison"); plt.ylabel("Accuracy"); plt.show()
