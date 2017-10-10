from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearchDBN, BicScore
from pgmpy.inference import DBNInference

import time
import pandas as pd
import random as rand
import networkx as nx

create = False

# CREATES SIMULATED DBN MODEL
dbn = DynamicBayesianNetwork()

#   Node    Name                Values
#   R       Robot Action        prompt, fail, reward
#   A       Subject Action      response, no response
#   O       Observation         Q values -> prompt, fail, reward
dbn.add_nodes_from(['R', 'A', 'O'])

# Check diagram for details
dbn.add_edges_from([(('R', 0), ('A', 0)),
                    (('R', 0), ('R', 1)),
                    (('A', 0), ('O', 0)),
                    (('A', 0), ('R', 1))])

# prompt r, nr
# fail   r, nr
# reward r, nr
cpd_R = TabularCPD(('R', 0), 3, [[1.0],
                                 [0.0],
                                 [0.0]])

# response      p,  f,  r
# no response   p,  f,  r
cpd_A = TabularCPD(('A', 0), 2, [[0.5, 0.0, 0.0],
                                 [0.5, 1.0, 1.0]],
                   [('R', 0)], [3])

# Q prompt r, nr
# Q fail   r, nr
# Q reward r, nr
cpd_O = TabularCPD(('O', 0), 3, [[0.15, 0.70],
                                 [0.05, 0.28],
                                 [0.80, 0.02]],
                   [('A', 0)], [2])

# prompt r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
# fail   r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
# reward r-p,  r-f,  r-r,  nr-p,  nr-f,  nr-r
cpd_R_1 = TabularCPD(('R', 1), 3,
                     [[0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.6, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
                     [('A', 0), ('R', 0)], [2, 3])

dbn.add_cpds(cpd_A, cpd_O, cpd_R, cpd_R_1)

dbn.initialize_initial_state()

print "Model created successfully: ", dbn.check_model()


# CREATES OR LOADS SIMULATED DATA FROM DBN MODEL
if create:
    # t1 = time.time()
    samples = dbn.create_samples(10000)
    # t = time.time() - t1
    # print t
    data = pd.DataFrame(samples)
    data.to_csv('../data/data.csv')
else:
    data = pd.DataFrame()
    data = data.from_csv('../data/data.csv')
    variables = data.columns.values
    if len(variables) > 0:
        nvars = list()
        for var in variables:
            nvar = var.replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').split(',')
            nvars.append((nvar[0], int(nvar[1])))
        data.columns = nvars

# LEARNS MODEL FROM SIMULATED DATA
hc = HillClimbSearchDBN(data, scoring_method=BicScore(data))
model = hc.estimate(tabu_length=10, max_indegree=2)
model.fit(data)
model.initialize_initial_state()

print "Model learned successfully: ", model.check_model()

print model.edges()
for cpd in model.get_cpds():
    print cpd

nx.drawing.nx_pydot.write_dot(model, "../output/network.dot")

dbn_infer = DBNInference(model)


def sample(values):
    index = 0
    rand_val = rand.random()
    freq = values[index]
    while freq < rand_val:
        index += 1
        freq += values[index]
    return index


t = 0
robot = 0
while True:
    print 'Robot (prompt=0, fail=1, reward=2): ', robot
    obs = input('Observation (prompt=0, fail=1, reward=2): ')
    q = dbn_infer.query(variables=[('A', t)], evidence={('O', t): obs,
                                                        ('R', t): robot})
    action = sample(q[('A', t)].values)
    print 'Action (resp=0, no resp=1): ', action
    q = dbn_infer.query(variables=[('R', t + 1)], evidence={('O', t): obs,
                                                            ('A', t): action,
                                                            ('R', t): robot})
    robot = sample(q[('R', t + 1)].values)

    if robot != 0:
        print 'Robot (prompt=0, fail=1, reward=2): ', robot
        print
        print
        print 'Reset'
        robot = 0
        t = 0
    else:
        t += 1
    print
