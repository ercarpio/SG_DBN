from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearchDBN, BicScore
import networkx as nx
import random as rand
import pandas as pd

# CREATES SIMULATED DBN MODEL

dbn = DynamicBayesianNetwork()

#   Node    Name                Values
#   I       Subject Interest    engaged, neutral, off
#   A       Subject Action      response, no response
#   R       Robot Action        prompt, fail, reward
#   O       Observation         q values
dbn.add_nodes_from(['I', 'A', 'R', 'O'])

# Check diagram for details
# I -----------> I2
# |  ------------^
# v /            |
# A ---> R -------
# |
# v
# O
dbn.add_edges_from([(('I', 0), ('A', 0)),
                    (('I', 0), ('R', 0)),
                    (('I', 0), ('I', 1)),
                    (('A', 0), ('O', 0)),
                    (('A', 0), ('R', 0)),
                    (('A', 0), ('I', 1)),
                    (('R', 0), ('I', 1))])

# engaged, neutral, off
cpd_I = TabularCPD(('I', 0), 3, [[0.3], [0.4], [0.3]])

# response    e, n, o
# no response e, n, o
cpd_A = TabularCPD(('A', 0), 2, [[0.9, 0.6, 0.1],
                                 [0.1, 0.4, 0.9]],
                   [('I', 0)], [3])

# prompt r-e, nr-e, r-n, nr-n, r-o, nr-o
# fail   r-e, nr-e, r-n, nr-n, r-o, nr-o
# reward r-e, nr-e, r-n, nr-n, r-o, nr-o
cpd_R = TabularCPD(('R', 0), 3, [[0.1, 0.9, 0.1, 0.9, 0.1, 0.1],
                                 [0.0, 0.1, 0.0, 0.1, 0.0, 0.9],
                                 [0.9, 0.0, 0.9, 0.0, 0.9, 0.0]],
                   [('I', 0), ('A', 0)], [3, 2])

# qval responsive r - nr
# qval non-resp   r - nr
cpd_O = TabularCPD(('O', 0), 2, [[0.8, 0.1],
                                 [0.2, 0.9]],
                   [('A', 0)], [2])

# engaged          r-e-p,   r-e-f,  r-e-r,  r-n-p,  r-n-f,  r-n-r,  r-o-p,  r-o-f,  r-o-r,
#                  nr-e-p, nr-e-f, nr-e-r, nr-n-p, nr-n-f, nr-n-r, nr-o-p, nr-o-f, nr-o-r,
# neutral
# off
cpd_I1 = TabularCPD(('I', 1), 3,
                    [[0.7, 0.1, 0.9, 0.6, 0.0, 0.8, 0.1, 0.0, 0.3, 0.5, 0.1, 0.7, 0.3, 0.0, 0.6, 0.1, 0.0, 0.1],
                     [0.2, 0.1, 0.1, 0.3, 0.2, 0.2, 0.3, 0.1, 0.5, 0.4, 0.1, 0.3, 0.4, 0.4, 0.4, 0.2, 0.1, 0.1],
                     [0.1, 0.8, 0.0, 0.1, 0.8, 0.0, 0.6, 0.9, 0.2, 0.1, 0.8, 0.0, 0.3, 0.6, 0.0, 0.7, 0.9, 0.8]],
                    [('A', 0), ('I', 0), ('R', 0)], [2, 3, 3])

dbn.add_cpds(cpd_I, cpd_A, cpd_I1, cpd_O, cpd_R)

dbn.initialize_initial_state()

print "Model created successfully: ", dbn.check_model()


# CREATES SIMULATED DATA FROM DBN MODEL

samples = list()

for i in range(1, 10000):
    top_order = list(nx.topological_sort(dbn))
    sample = dict()
    for node in top_order:
        curr_cpd = dbn.get_cpds(node)
        evidence = curr_cpd.get_evidence()
        ev_index = 0
        if len(evidence) != 0:
            for var in evidence:
                curr_cpd = curr_cpd.reduce([(var, sample[var])], inplace=False)
        row_index = 0
        values = curr_cpd.get_values()
        rand_val = rand.random()
        freq = values[row_index][ev_index]
        while freq < rand_val:
            row_index += 1
            freq += values[row_index][ev_index]
        sample[node] = row_index
        # print values
        # print node, row_index
        if node == ('O', 0):
            break
    samples.append(sample)


# LEARNS MODEL FROM SIMULATED DATA

data = pd.DataFrame(samples)
data.columns = ['A', 'I', 'I_1', 'O', 'R']

hc = HillClimbSearchDBN(data, scoring_method=BicScore(data))
model = hc.estimate(tabu_length=0)
model.fit(data)

print model.edges()
for cpd in model.get_cpds():
    print cpd
