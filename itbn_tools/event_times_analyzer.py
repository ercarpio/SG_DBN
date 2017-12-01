import os
import numpy as np

EVENT_LABEL_IX = 0
VALUE_IX = 1

sessions = dict()
# events that are found in the label files
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
# create an empty array
counter = 0
data_array = np.full((len(sessions), len(actions_dict)), -1.0)
for file, events in sessions.items():
    for event, value in events.items():
        data_array[counter][actions_dict[event]] = value
    counter += 1
print(data_array)
