import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
X,y=load_iris(return_X_y=True); a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
s=StandardScaler(); a=s.fit_transform(a); b=s.transform(b)
m=keras.Sequential([keras.layers.Input((4,)),keras.layers.Dense(16,activation="relu"),keras.layers.Dense(8,activation="relu"),keras.layers.Dense(3,activation="softmax")])
m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
h=m.fit(a,c,epochs=40,validation_split=.2,verbose=0); print("Test metrics:",m.evaluate(b,d,verbose=0))
plt.plot(h.history["accuracy"],label="Train"); plt.plot(h.history["val_accuracy"],label="Validation"); plt.legend(); plt.title("MLP Accuracy"); plt.show()
