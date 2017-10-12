from pgmpy.estimators import HillClimbSearchDBN, BicScore
from pgmpy.inference import DBNInference
from pgmpy.models import DynamicBayesianNetwork

import pandas as pd
import numpy as np
import networkx as nx


def sample(values):
    index = 0
    rand_val = np.random.rand(1, 1)[0]
    freq = values[index]
    while freq < rand_val:
        index += 1
        freq += values[index]
    return index


data = pd.DataFrame()
data = data.from_csv('../data/simple_qval_data.csv')
variables = data.columns.values
if len(variables) > 0:
    nvars = list()
    for var in variables:
        nvar = var.replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').split(',')
        nvars.append((nvar[0], int(nvar[1])))
    data.columns = nvars


# LEARNS MODEL FROM REAL DATA
hc = HillClimbSearchDBN(data, scoring_method=BicScore(data))
print 'Learning model'

# GIVE STRUCTURE LEARNING ALGORITHM A HINT OF THE MODEL
nodes = hc.state_names.keys()
start = DynamicBayesianNetwork()
nodes = set(X[0] for X in nodes)
start.add_nodes_from_ts(nodes, [0, 1])
# start.add_edge(('A', 0), ('A', 1))
# start.add_edge(('R', 0), ('R', 1))
start.add_edge(('P', 0), ('P', 1))

model = hc.estimate(start=start, tabu_length=10, max_indegree=2)
# model = hc.estimate(tabu_length=5, max_indegree=2)
print 'Learning parameters'
model.fit(data)
# model.fit(data, estimator=BayesianEstimator)
model.initialize_initial_state()
print "Model learned successfully: ", model.check_model()

print model.edges()
for cpd in model.get_cpds():
    print cpd

# DRAWS RESULTING NETWORK
nx.drawing.nx_pydot.write_dot(model, "../output/network.dot")

dbn_infer = DBNInference(model)

t = 0
prompt = 0
reward = 0
failure = 0
while True:
    print prompt, reward, failure
    q = dbn_infer.query(variables=[('P', t + 1), ('R', t + 1), ('A', t + 1)], evidence={('P', t): prompt,
                                                                                        ('R', t): reward,
                                                                                        ('A', t): failure})
    print q.values()
    obs = input('Observation (prompt=0, fail=1, reward=2): ')
    if obs == 0:
        prompt = 1
    elif obs == 1:
        failure = 1
    elif obs == 2:
        reward = 1
    if robot != 0:
        print
        print 'Reset'
        t = 0
        prompt = 0
        reward = 0
        failure = 0
    else:
        t += 1
    print

# https://codeyarns.com/2013/06/24/how-to-convert-dot-file-to-image-format/
# https://stackoverflow.com/questions/10379448/plotting-directed-graphs-in-python-in-a-way-that-show-all-edges-separately
