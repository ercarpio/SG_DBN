from pgmpy.models import IntervalTemporalBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
import pandas as pd
import numpy as np

raw = np.random.randint(0, 1, size=(10, 6))
raw[:, 0] = np.random.randint(0, 4)
raw[:, 1] = np.random.randint(2, 7)
raw[:, 2] = raw[:, 0] + np.random.randint(1, 4)
raw[:, 3] = raw[:, 1] + np.random.randint(1, 4)
raw[:, 4] = np.random.randint(0, 2)

data = pd.DataFrame(raw, columns=['C_s', 'W_s', 'C_e', 'W_e', 'C', 'W'])

hc = HillClimbSearch(data, scoring_method=BicScore(data))

start = IntervalTemporalBayesianNetwork()
model = hc.estimate(start=start, tabu_length=10, max_indegree=3)
print model.edges()

model.learn_temporal_relationships(data)
model.add_temporal_nodes()
print model.edges()

model.fit(data[['C', 'W', 't_C_W']])
for cpd in model.get_cpds():
    print cpd
