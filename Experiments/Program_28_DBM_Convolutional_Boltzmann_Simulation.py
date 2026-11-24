import numpy as np, matplotlib.pyplot as plt
from scipy.special import expit
rng=np.random.default_rng(4); v=np.array([1.,0.,1.,0.]); W1=rng.normal(0,.2,(4,3)); W2=rng.normal(0,.2,(3,2))
h1=(rng.random(3)<expit(v@W1)).astype(float); h2=(rng.random(2)<expit(h1@W2)).astype(float)
h1b=(rng.random(3)<expit(h2@W2.T)).astype(float); pv=expit(h1b@W1.T)
print("Visible:",v,"Hidden layer 1:",h1,"Hidden layer 2:",h2)
print("Reconstructed hidden:",h1b,"Visible probabilities:",pv)
plt.bar(range(4),pv); plt.ylim(0,1); plt.title("Layered Boltzmann-Style Reconstruction"); plt.show()
print("Compact educational simulation, not a full-scale trained DBM/ConvBM.")
