import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
X,y=make_moons(n_samples=1000,noise=.3,random_state=2); a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42)
s=StandardScaler(); a=s.fit_transform(a); b=s.transform(b)
for name,reg,drop,early in [("Baseline",None,False,False),("L2",keras.regularizers.l2(.01),False,False),("Dropout",None,True,False),("Early stop",None,False,True)]:
 layers=[keras.layers.Input((2,)),keras.layers.Dense(48,activation="relu",kernel_regularizer=reg)]
 if drop: layers.append(keras.layers.Dropout(.4))
 layers += [keras.layers.Dense(24,activation="relu"),keras.layers.Dense(1,activation="sigmoid")]
 m=keras.Sequential(layers); m.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
 cb=[keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)] if early else []
 h=m.fit(a,c,epochs=50,validation_split=.2,callbacks=cb,verbose=0)
 print(name,m.evaluate(b,d,verbose=0)); plt.plot(h.history["val_loss"],label=name)
plt.title("Regularization Comparison"); plt.xlabel("Epoch"); plt.ylabel("Validation loss"); plt.legend(); plt.show()
