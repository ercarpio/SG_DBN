from pgmpy.models import IntervalTemporalBayesianNetwork
import pandas as pd
import numpy as np

model = IntervalTemporalBayesianNetwork()
model.add_node("Command")
model.add_node("Wave")
model.add_edge("Command", 'Wave')

raw = np.random.randint(0, 4, 30 * 6)
raw = np.reshape(raw, [-1, 6])
raw[:, 2] = raw[:, 0] + 1
raw[:, 3] = raw[:, 1] + 1
data = pd.DataFrame(raw, columns=['Command_s', 'Wave_s', 'Command_e', 'Wave_e', 'Command', 'Wave'])

model.learn_temporal_relationships(data)
print model.edges()
model.add_temporal_nodes()
print model.edges()
model.fit(data[['Command', 'Wave', 't_Command_Wave']])
for cpd in model.get_cpds():
    print cpd
