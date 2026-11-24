import numpy as np, matplotlib.pyplot as plt
from scipy.special import expit
rng=np.random.default_rng(1); v=np.array([1.,0.,1.,0.]); W=rng.normal(0,.3,(4,3)); vb=np.zeros(4); hb=np.zeros(3)
ph=expit(v@W+hb); h=(rng.random(3)<ph).astype(float); pv=expit(h@W.T+vb); vr=(rng.random(4)<pv).astype(float)
E=-v@vb-h@hb-v@W@h
print("Input:",v,"Hidden probabilities:",ph,"Hidden sample:",h)
print("Reconstruction probabilities:",pv,"Reconstructed sample:",vr,"Energy:",E)
plt.bar(range(4),pv); plt.ylim(0,1); plt.title("RBM Reconstruction"); plt.show()
