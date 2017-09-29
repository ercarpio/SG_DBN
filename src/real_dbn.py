from pgmpy.estimators import HillClimbSearchDBN, BicScore, BdeuScore, BayesianEstimator
from pgmpy.inference import DBNInference
from pgmpy.models import DynamicBayesianNetwork as dbn

import pandas as pd
import random as rand

data = pd.DataFrame()
data = data.from_csv('../data/qval_data.csv')
variables = data.columns.values
if len(variables) > 0:
    nvars = list()
    for var in variables:
        nvar = var.replace('(', '').replace(')', '').replace('\'', '').replace(' ', '').split(',')
        nvars.append((nvar[0], int(nvar[1])))
    data.columns = nvars


# LEARNS MODEL FROM REAL DATA
hc = HillClimbSearchDBN(data, scoring_method=BicScore(data))
# hc = HillClimbSearchDBN(data, scoring_method=BdeuScore(data))
print 'Learning model'
model = hc.estimate(tabu_length=5, max_indegree=2)
print 'Learning parameters'
model.fit(data, estimator=BayesianEstimator)
# model.fit(data, estimator=BayesianEstimator)
model.initialize_initial_state()

print "Model learned successfully: ", model.check_model()

print model.edges()
for cpd in model.get_cpds():
    print cpd
#
# dbn_infer = DBNInference(model)
#
#
# def sample(values):
#     index = 0
#     rand_val = rand.random()
#     freq = values[index]
#     while freq < rand_val:
#         index += 1
#         freq += values[index]
#     return index
#
#
# t = 0
# robot = 0
# while True:
#     print 'Robot (prompt=0, fail=1, reward=2): ', robot
#     obs = input('Observation (prompt=0, fail=1, reward=2): ')
#     q = dbn_infer.query(variables=[('A', t)], evidence={('O', t): obs,
#                                                         ('R', t): robot})
#     action = sample(q[('A', t)].values)
#     print 'Action (resp=0, no resp=1): ', action
#     q = dbn_infer.query(variables=[('R', t + 1)], evidence={('O', t): obs,
#                                                             ('A', t): action,
#                                                             ('R', t): robot})
#     robot = sample(q[('R', t + 1)].values)
#
#     if robot != 0:
#         print 'Robot (prompt=0, fail=1, reward=2): ', robot
#         print
#         print
#         print 'Reset'
#         robot = 0
#         t = 0
#     else:
#         t += 1
#     print
