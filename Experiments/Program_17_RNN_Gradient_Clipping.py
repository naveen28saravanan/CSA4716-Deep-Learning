import numpy as np, matplotlib.pyplot as plt
g=np.array([12.,-8.,5.,-15.]); limit=5.; norm=np.linalg.norm(g)
clipped=g*min(1,limit/(norm+1e-12))
print("Original:",g,"norm:",norm); print("Clipped:",clipped,"norm:",np.linalg.norm(clipped))
t=np.arange(1,16)
for r in [.5,.9,1.1,1.5]: plt.plot(t,np.abs(r)**t,label=f"factor {r}")
plt.yscale("log"); plt.title("Vanishing and Exploding Gradients"); plt.xlabel("Recurrent steps"); plt.legend(); plt.show()
