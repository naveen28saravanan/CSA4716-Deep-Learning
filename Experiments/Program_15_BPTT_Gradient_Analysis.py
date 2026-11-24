import numpy as np, matplotlib.pyplot as plt
x=[.5,-.2,.8,.1,.4]; w=.7; u=.9
def run(weight):
 h=0.; states=[]
 for v in x: h=np.tanh(weight*h+u*v); states.append(h)
 return h,states
value,states=run(w); eps=1e-5; grad=(run(w+eps)[0]-run(w-eps)[0])/(2*eps)
print("Hidden states:",states); print("Approximate gradient dh_final/dw:",grad)
plt.plot(range(1,6),1-np.array(states)**2,marker="o"); plt.title("BPTT Local Gradient Factors"); plt.xlabel("Time step"); plt.show()
