from itertools import permutations

from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearchDBN, BicScore

import time
import pandas as pd


# CREATES SIMULATED DBN MODEL

dbn = DynamicBayesianNetwork()

#   Node    Name                Values
#   A       Subject Action      response, no response
#   R       Robot Action        prompt, fail, reward
#   O       Observation         Q values -> prompt, fail, reward
dbn.add_nodes_from(['A', 'R', 'O'])

# Check diagram for details
dbn.add_edges_from([(('A', 0), ('O', 0)),
                    (('A', 0), ('R', 0)),
                    (('A', 0), ('A', 1)),
                    (('R', 0), ('A', 1)),
                    (('R', 0), ('R', 1))])

# response
# no response
cpd_A = TabularCPD(('A', 0), 2, [[0.5],
                                 [0.5]])

# prompt r, nr
# fail   r, nr
# reward r, nr
cpd_R = TabularCPD(('R', 0), 3, [[0.0, 0.9],
                                 [0.0, 0.1],
                                 [1.0, 0.0]],
                   [('A', 0)], [2])

# Q prompt r, nr
# Q fail   r, nr
# Q reward r, nr
cpd_O = TabularCPD(('O', 0), 3, [[0.15, 0.70],
                                 [0.05, 0.28],
                                 [0.80, 0.02]],
                   [('A', 0)], [2])

# response    r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
# no response r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
cpd_A_1 = TabularCPD(('A', 1), 2,
                     [[0.9, 0.0, 0.0, 0.6, 0.0, 0.0],
                      [0.1, 1.0, 1.0, 0.4, 1.0, 1.0]],
                     [('A', 0), ('R', 0)], [2, 3])

# prompt r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
# fail   r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
# reward r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
cpd_R_1 = TabularCPD(('R', 1), 3,
                     [[0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.7, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
                     [('A', 1), ('R', 0)], [2, 3])

dbn.add_cpds(cpd_A, cpd_A_1, cpd_O, cpd_R, cpd_R_1)

dbn.initialize_initial_state()

print "Model created successfully: ", dbn.check_model()


# CREATES SIMULATED DATA FROM DBN MODEL
t1 = time.time()
samples = dbn.create_samples(10000)
t = time.time() - t1
print t

# LEARNS MODEL FROM SIMULATED DATA

data = pd.DataFrame(samples)

hc = HillClimbSearchDBN(data, scoring_method=BicScore(data))
model = hc.estimate(tabu_length=0)
model.fit(data)

print model.edges()
for cpd in model.get_cpds():
    print cpd
