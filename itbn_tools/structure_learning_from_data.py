import os
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.models import IntervalTemporalBayesianNetwork as ITBN
from pgmpy.estimators import HillClimbSearchITBN, BicScore

EVENT_LABEL_IX = 0
VALUE_IX = 1

sessions = dict()
# events that are found in the label files
actions_dict = {'command_s': 0,     'command_e': 1,     'command': 2,
                'noise_0_s': 0,     'noise_0_e': 1,     'noise_0': 2,
                'prompt_s': 3,      'prompt_e': 4,      'prompt': 5,
                'noise_1_s': 3,     'noise_1_e': 4,     'noise_1': 5,
                'reward_s': 6,      'reward_e': 7,      'reward': 8,
                'abort_s': 9,       'abort_e': 10,      'abort': 11,
                'audio_0_s': 12,    'audio_0_e': 13,    'audio_0': 14,
                'audio_1_s': 12,    'audio_1_e': 13,    'audio_1': 14,
                'gesture_0_s': 15,  'gesture_0_e': 16,  'gesture_0': 17,
                'gesture_1_s': 15,  'gesture_1_e': 16,  'gesture_1': 17}
# some sessions need to be corrected to deliver a reward after a correct response
files_to_shorten = {'01': ['a0', 'g0', 'ga0', 'za0', 'zg0', 'zga0'],
                    '02': ['a0', 'g0', 'ga0', 'za0', 'zg0', 'zga0'],
                    '03': ['a0', 'ga0', 'za0', 'zg0', 'zga0'],
                    '04': ['a0', 'g0', 'ga0', 'za0', 'zg0', 'zga0']}
# some sessions need to be corrected to remove a correct response before the prompt
files_to_correct = {'01': ['a1', 'g1', 'ga1', 'za1', 'zg1', 'zga1'],
                    '02': ['a1', 'g1', 'ga1', 'za1', 'zg1', 'zga1']}

# go over all the files in the label directory
for root, subFolders, files in os.walk('../labels/'):
    for f in files:
        shorten = False
        correct = False
        n_times = dict()
        # open each file and read line by line
        file_path = os.path.join(root, f)
        session_dict = dict()
        session_file = open(file_path, 'r')
        # check if the session needs to be shortened
        if f.replace('.txt', '') in files_to_shorten.get(root.split('_')[1], []):
            shorten = True
        # check if the session needs to be corrected
        elif f.replace('.txt', '') in files_to_correct.get(root.split('_')[1], []):
            correct = True
        for line in session_file:
            # assign the reward_s the time of prompt_s, fix reward_e, abort_s, abort_e
            # accordingly and remove times of prompt, audio_1 and gesture_1
            if shorten:
                if 'prompt_s' in line:
                    n_times['prompt_s'] = float(line.split(' ')[VALUE_IX])
                    continue
                elif '_1' in line or 'prompt' in line:
                    continue
                elif 'reward_s' in line:
                    n_times['reward_s'] = float(line.split(' ')[VALUE_IX])
                    line = 'reward_s ' + str(n_times['prompt_s'])
                elif 'abort' in line or 'reward' in line:
                    line = line.split(' ')[EVENT_LABEL_IX] + ' ' + str(n_times['prompt_s'] +
                                                                (float(line.split(' ')[VALUE_IX]) -
                                                                 n_times['reward_s']))
            # remove the times of audio_0 and gesture_0
            elif correct:
                if '_0' in line:
                    continue
            data = line.split(' ')
            session_dict[data[EVENT_LABEL_IX]] = float(data[VALUE_IX])
        sessions[file_path] = session_dict

# create an empty array to store the final data
counter = 0
data_array = np.full((len(sessions), len(actions_dict) - 6), -1.0)
for file, events in sessions.items():
    for event, value in events.items():
        data_array[counter][actions_dict[event]] = value
        if event.endswith(ITBN.start_time_marker):
            data_array[counter][actions_dict[event.replace(ITBN.start_time_marker, '')]] = 1
    counter += 1

# Create DataFrame to hold data
data = pd.DataFrame(data_array, columns=['command_s', 'command_e', 'command',
                                         'prompt_s', 'prompt_e', 'prompt',
                                         'reward_s', 'reward_e', 'reward',
                                         'abort_s', 'abort_e', 'abort',
                                         'audio_s', 'audio_e', 'audio',
                                         'gesture_s', 'gesture_e', 'gesture'])

# Create empty model and add event nodes
model = ITBN()
model.add_nodes_from(data.columns.values)

# Learn temporal relations from data
model.learn_temporal_relationships(data)

# Delete columns with temporal information
data.fillna(0, inplace=True)
for col in list(data.columns.values):
    if col.endswith(ITBN.start_time_marker) or col.endswith(ITBN.end_time_marker):
        data.drop(col, axis=1, inplace=True)

# Learn model structure from data and temporal relations
hc = HillClimbSearchITBN(data, scoring_method=BicScore(data))
model = hc.estimate(start=model)

# Learn model parameters
# model.fit(list(data[model.nodes()]))
for cpd in model.get_cpds():
    print(cpd)

# Draws and outputs resulting network
model.draw_to_file("../output/itbn.png")
os.system('gnome-open ../output/itbn.png')
nx.write_gpickle(model, "../output/itbn.nx")
