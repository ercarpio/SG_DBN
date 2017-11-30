import os
import numpy as np

EVENT_LABEL_IX = 0
VALUE_IX = 1

sessions = dict()
actions_dict = {'command_s': 0,
                'command_e': 1,
                'prompt_s': 2,
                'prompt_e': 3,
                'reward_s': 4,
                'reward_e': 5,
                'abort_s': 6,
                'abort_e': 7,
                'audio_0_s': 8,
                'audio_0_e': 9,
                'audio_1_s': 10,
                'audio_1_e': 11,
                'gesture_0_s': 12,
                'gesture_0_e': 13,
                'gesture_1_s': 14,
                'gesture_1_e': 15}
for root, subFolders, files in os.walk('../labels/'):
    for f in files:
        file_path = os.path.join(root, f)
        # print('Processing: {}'.format(file_path))
        session_dict = dict()
        session_file = open(file_path, 'r')
        for line in session_file:
            data = line.split(' ')
            session_dict[data[EVENT_LABEL_IX]] = float(data[VALUE_IX])
        sessions[file_path] = session_dict
data_array = np.full((len(sessions), len(actions_dict)), -1.0)
counter = 0
for file, events in sessions.items():
    for event, value in events.items():
        data_array[counter][actions_dict[event]] = value
    counter += 1
    # print(data_array)
print(data_array)
