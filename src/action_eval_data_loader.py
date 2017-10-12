from pgmpy.estimators import HillClimbSearchDBN, BicScore
from pgmpy.inference import DBNInference
from pgmpy.models import DynamicBayesianNetwork

import pandas as pd
import networkx as nx
import numpy as np

IX_PAST_ACTION = 0
IX_Q_VALUES = 1
PROMPT = 0
REWARD = 1
ABORT = 2
IX_CORRECT_ACTION = 2

# LOADS RAW DATA
with open('../data/action_eval', 'r') as f:
    lines = f.readlines()
    samples = dict()
    counts = np.ndarray(shape=(3,), dtype=float)
    counts.fill(0)
    for i in range(0, len(lines)):
        line = lines[i]
        if line.startswith('final'):
            # REMOVE UNWANTED CHARS
            line = line.replace('\n', '').replace('[', '').replace(']', '')
            while line != line.replace('  ', ''):
                line = line.replace('  ', ' ')
            fields = line.split(' ')
            # STORE VALUABLE INFORMATION
            samples[fields[0]] = [int(float(fields[1])),
                                  [float(fields[2]), float(fields[3]), float(fields[4])],
                                  int(fields[5])]
            counts[int(fields[5])] = counts[int(fields[5])] + 1
    print 'Data loaded. Total samples: {0}'.format(sum(counts))


# PREPARES DATA FOR DBN
final_samples = list()
for key in samples.keys():
    values = samples[key]
    last_action = values[IX_PAST_ACTION]
    curr_sample = dict()
    curr_sample[('Prompt', 0)] = 0
    curr_sample[('Reward', 0)] = 0
    curr_sample[('Abort', 0)] = 0
    curr_sample[('Prompt', 1)] = 0
    curr_sample[('Reward', 1)] = 0
    curr_sample[('Abort', 1)] = 0
    if last_action == 1:
        curr_sample[('Prompt', 0)] = 1
    if values[IX_CORRECT_ACTION] == PROMPT:
        curr_sample[('Prompt', 1)] = 1
    elif values[IX_CORRECT_ACTION] == REWARD:
        curr_sample[('Reward', 1)] = 1
    elif values[IX_CORRECT_ACTION] == ABORT:
        curr_sample[('Abort', 1)] = 1
    final_samples.append(curr_sample)


# LEARNS STRUCTURE FROM DATA
print 'Learning model'
data = pd.DataFrame(final_samples)
hc = HillClimbSearchDBN(data, scoring_method=BicScore(data))
# GIVE STRUCTURE LEARNING ALGORITHM A HINT OF THE STRUCTURE
nodes = hc.state_names.keys()
start = DynamicBayesianNetwork()
nodes = set(X[0] for X in nodes)
start.add_nodes_from_ts(nodes, [0, 1])
# start.add_edge(('P', 0), ('R', 0))
# start.add_edge(('P', 0), ('R', 1))
# start.add_edge(('P', 0), ('A', 0))
# start.add_edge(('P', 0), ('A', 1))
# start.add_edge(('P', 0), ('P', 1))
model = hc.estimate(start=start, tabu_length=10, max_indegree=2)

# LEARNS PARAMETERS FROM DATA
print 'Learning parameters'
model.fit(data)
# model.fit(data, estimator=BayesianEstimator)

# FINALIZES MODEL
model.initialize_initial_state()
print "Model learned successfully: ", model.check_model()

# PRINTS MODEL
print model.edges()
for cpd in model.get_cpds():
    print cpd

# DRAWS RESULTING NETWORK
nx.drawing.nx_pydot.write_dot(model, "../output/network.dot")

# OUTPUTS RESULTING NETWORK
nx.write_gpickle(model, "../output/network.nx")

# TESTS MODEL
dbn_infer = DBNInference(model)
correct_prob_actions = np.ndarray(shape=(3,), dtype=int)
correct_prob_actions.fill(0)
correct_binary_actions = np.ndarray(shape=(3,), dtype=int)
correct_binary_actions.fill(0)
inferred_prob = np.ndarray(shape=(3,), dtype=float)
inferred_binary = np.ndarray(shape=(3,), dtype=float)
for sample in samples.keys():
    inferred_prob.fill(0)
    inferred_binary.fill(0)
    values = samples[sample]
    variables = (('Prompt', 1), ('Reward', 1), ('Abort', 1))
    evidence = {('Prompt', 0): 0, ('Reward', 0): 0, ('Abort', 0): 0}
    if values[IX_PAST_ACTION] == 1:
        evidence[('Prompt', 0)] = 1
    q = dbn_infer.query(variables=variables, evidence=evidence)
    for variable in q.values():
        action = variables.index(variable.variables[0])
        inferred_prob[action] = variable.values[1]
        inferred_binary[action] = 1 if variable.values[1] > 0 else 0
    revised_prob = np.argmax(values[IX_Q_VALUES] * inferred_prob)
    revised_binary = np.argmax(values[IX_Q_VALUES] * inferred_binary)
    if revised_prob == values[IX_CORRECT_ACTION]:
        correct_prob_actions[revised_prob] = correct_prob_actions[revised_prob] + 1
    if revised_binary == values[IX_CORRECT_ACTION]:
        correct_binary_actions[revised_binary] = correct_binary_actions[revised_binary] + 1
print 'PROBABILITY METHOD\nTotal Accuracy: {:10.5f} ({:}/{:})\n' \
      'Prompt Accuracy: {:10.5f} ({:}/{:})\n' \
      'Reward Accuracy: {:10.5f} ({:}/{:})\n' \
      'Abort Accuracy: {:10.5f} ({:}/{:})\n'.format(
        sum(correct_prob_actions) / sum(counts),
        sum(correct_prob_actions), sum(counts),
        correct_prob_actions[PROMPT] / counts[PROMPT],
        correct_prob_actions[PROMPT], counts[PROMPT],
        correct_prob_actions[REWARD] / counts[REWARD],
        correct_prob_actions[REWARD], counts[REWARD],
        correct_prob_actions[ABORT] / counts[ABORT],
        correct_prob_actions[ABORT], counts[ABORT])
print 'BINARY METHOD\nTotal Accuracy: {:10.5f} ({:}/{:})\n' \
      'Prompt Accuracy: {:10.5f} ({:}/{:})\n' \
      'Reward Accuracy: {:10.5f} ({:}/{:})\n' \
      'Abort Accuracy: {:10.5f} ({:}/{:})\n'.format(
        sum(correct_binary_actions) / sum(counts),
        sum(correct_binary_actions), sum(counts),
        correct_binary_actions[PROMPT] / counts[PROMPT],
        correct_binary_actions[PROMPT], counts[PROMPT],
        correct_binary_actions[REWARD] / counts[REWARD],
        correct_binary_actions[REWARD], counts[REWARD],
        correct_binary_actions[ABORT] / counts[ABORT],
        correct_binary_actions[ABORT], counts[ABORT])
