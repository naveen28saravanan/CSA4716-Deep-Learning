import networkx as nx, matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
m=DiscreteBayesianNetwork([("Cloudy","Rain"),("Cloudy","Sprinkler"),("Rain","WetGrass"),("Sprinkler","WetGrass")])
m.add_cpds(TabularCPD("Cloudy",2,[[.5],[.5]]),
TabularCPD("Rain",2,[[.8,.2],[.2,.8]],evidence=["Cloudy"],evidence_card=[2]),
TabularCPD("Sprinkler",2,[[.5,.9],[.5,.1]],evidence=["Cloudy"],evidence_card=[2]),
TabularCPD("WetGrass",2,[[1,.1,.1,.01],[0,.9,.9,.99]],evidence=["Rain","Sprinkler"],evidence_card=[2,2]))
print("Valid model:",m.check_model()); print(VariableElimination(m).query(["Rain"],evidence={"WetGrass":1}))
G=nx.DiGraph(m.edges()); nx.draw_networkx(G,with_labels=True,node_color="lightblue",node_size=1800); plt.show()
