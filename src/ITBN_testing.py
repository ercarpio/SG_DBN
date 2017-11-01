from pgmpy.models import IntervalTemporalBayesianNetwork
import pandas as pd
import numpy as np

model = IntervalTemporalBayesianNetwork()
model.add_node("Command")
model.add_node("Wave")
model.add_node("t_command_wave")

raw = np.random.rand(30, 2)
data = pd.DataFrame(raw, columns=['Command', 'Wave'])

model.learn_temporal_relationships(data)
