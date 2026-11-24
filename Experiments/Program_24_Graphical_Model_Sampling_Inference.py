import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import VariableElimination
m=DiscreteBayesianNetwork([("Rain","WetGrass"),("Sprinkler","WetGrass")])
m.add_cpds(TabularCPD("Rain",2,[[.8],[.2]]),TabularCPD("Sprinkler",2,[[.6],[.4]]),
TabularCPD("WetGrass",2,[[1,.2,.1,.01],[0,.8,.9,.99]],evidence=["Rain","Sprinkler"],evidence_card=[2,2]))
samples=BayesianModelSampling(m).forward_sample(size=3000,show_progress=False)
print(samples.head()); print("Sample P(WetGrass=1):",samples.WetGrass.mean())
print(VariableElimination(m).query(["Rain"],evidence={"WetGrass":1}))
samples.WetGrass.value_counts(normalize=True).plot(kind="bar"); plt.title("Sampled Wet Grass States"); plt.show()
