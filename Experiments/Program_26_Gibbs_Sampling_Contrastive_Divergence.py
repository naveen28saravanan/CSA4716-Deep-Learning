import numpy as np, matplotlib.pyplot as plt
from scipy.special import expit
rng=np.random.default_rng(2); data=np.array([[1,0,1,0],[1,1,0,0],[0,1,1,0],[0,0,1,1]],float)
W=rng.normal(0,.1,(4,2)); vb=np.zeros(4); hb=np.zeros(2); errors=[]
for epoch in range(200):
 for v in data:
  ph=expit(v@W+hb); h=(rng.random(2)<ph).astype(float)
  pv=expit(h@W.T+vb); vk=(rng.random(4)<pv).astype(float); ph1=expit(vk@W+hb)
  W+=.05*(np.outer(v,ph)-np.outer(vk,ph1)); vb+=.05*(v-vk); hb+=.05*(ph-ph1)
 errors.append(np.mean((data-expit(expit(data@W+hb)@W.T+vb))**2))
print("Final reconstruction error:",errors[-1]); plt.plot(errors); plt.title("Contrastive Divergence"); plt.show()
