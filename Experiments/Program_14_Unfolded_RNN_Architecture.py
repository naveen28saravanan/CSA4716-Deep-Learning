import networkx as nx, matplotlib.pyplot as plt
G=nx.DiGraph()
for t in range(5):
 G.add_node(f"x{t}",pos=(t,0)); G.add_node(f"h{t}",pos=(t,1)); G.add_node(f"y{t}",pos=(t,2))
 G.add_edge(f"x{t}",f"h{t}"); G.add_edge(f"h{t}",f"y{t}")
 if t: G.add_edge(f"h{t-1}",f"h{t}")
pos=nx.get_node_attributes(G,"pos"); nx.draw_networkx(G,pos,node_size=1300,node_color="lightblue",arrows=True)
plt.title("Unfolded RNN Across Time"); plt.axis("off"); plt.show()
