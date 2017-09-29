import pandas as pd

# LOADS RAW DATA
with open('../data/q-values.txt', 'r') as f:
    lines = f.readlines()
    recordings = dict()
    count = 0
    count_repeated = 0
    count_discarded = 0

    for i in range(0, len(lines)):
        line = lines[i]
        if line.startswith('noncompliant') or line.startswith('compliant'):
            line = line.replace('\n', '').replace('_flip', '100')
            if recordings.get(line, None) is None:
                count += 1
                recordings[line] = lines[i + 1].replace('\n', '').replace('\t', ' ')
            elif recordings[line] != lines[i + 1].replace('\n', '').replace('\t', ' '):
                count_repeated += 1
                recordings[line + 'r'] = lines[i + 1].replace('\n', '').replace('\t', ' ')
            else:
                count_discarded += 1
    print 'Data loaded.\nTotal:\t\t{0}\nRepeated:\t{1}\nDiscarded:\t{2}\n'.format(
        len(recordings.keys()), count_repeated, count_discarded)

# FIRST SWEPT, DEALS WITH FLIPS AND IDENTIFIES BROKEN SAMPLES
samples = dict()
for key in sorted(recordings.keys()):
    keys = key.split('_')
    if not keys[3].endswith('r'):
        samples[(keys[0], int(keys[1]), int(keys[2]), int(keys[3]))] = recordings[key]
    else:
        samples[(keys[0], int(keys[1]), -int(keys[2]), int(keys[3].replace('r', '')))] = recordings[key]


# SECOND SWEPT REPAIRS BROKEN SAMPLES, NUMBERS FINAL SAMPLES
final_samples = dict()
sample_count = -1
last_id = 0
for key in sorted(samples.keys()):
    if key[3] == 0:
        sample_count += 1
        final_samples[(key[0], sample_count, key[3])] = samples[key]
        last_id = key[2]
    elif key[2] == last_id:
        final_samples[(key[0], sample_count, key[3])] = samples[key]
    else:
        if samples.get((key[0], key[1], -key[2], 0), None) is not None:
            sample_count += 1
            final_samples[(key[0], sample_count, 0)] = samples.get((key[0], key[1], -key[2], 0), None)
            final_samples[(key[0], sample_count, key[3])] = samples[key]
            last_id = key[2]
        else:
            print 'error', key


# THIRD SWEPT FIXES INCONSISTENCIES
for key in final_samples.keys():
    if key[0] == 'noncompliant':
        if key[2] == 0:
            if final_samples.get((key[0], key[1], 1), None) is None:
                # print final_samples[(key[0], key[1] + 10, 1)]
                final_samples[(key[0], key[1], 1)] = final_samples[(key[0], key[1] + 10, 1)]
    else:
        if key[2] == 0:
            if final_samples.get((key[0], key[1], 1), None) is None:
                if final_samples[key].endswith('1'):
                    # print final_samples[(key[0], 3, 1)]
                    final_samples[(key[0], key[1], 1)] = final_samples[(key[0], 3, 1)]

# FOURTH SWEPT PREPARES SAMPLES FOR OUTPUT
curr_samples = dict()
out_samples = list()
prev_obs = 0


def get_obs(value):
    obs = value
    if value == '1':
        obs = 0
    elif value == '3':
        obs = 1
    return obs


for key in sorted(final_samples.keys()):
    sample_values = final_samples[key].split(' ')
    obs = get_obs(sample_values[len(sample_values) - 2])
    if key[0] == 'compliant' and final_samples[key].endswith('2'):
        if final_samples.get((key[0], key[1], 1), None) is None:
            curr_sample = dict()
            curr_sample[('A', 0)] = 1
            curr_sample[('O', 0)] = 0
            curr_sample[('R', 0)] = 0
            curr_sample[('A', 1)] = 0
            curr_sample[('O', 1)] = obs
            curr_sample[('R', 1)] = 2
            out_samples.append(curr_sample)
        else:
            curr_sample = dict()
            curr_sample[('A', 0)] = 1
            curr_sample[('O', 0)] = prev_obs
            curr_sample[('R', 0)] = 0
            curr_sample[('A', 1)] = 0
            curr_sample[('O', 1)] = obs
            curr_sample[('R', 1)] = 2
            out_samples.append(curr_sample)
    elif final_samples[key].endswith('1'):
        curr_sample = dict()
        curr_sample[('A', 0)] = 1
        curr_sample[('O', 0)] = 0
        curr_sample[('R', 0)] = 0
        curr_sample[('A', 1)] = 1
        curr_sample[('O', 1)] = obs
        curr_sample[('R', 1)] = 0
        prev_obs = obs
        out_samples.append(curr_sample)
    elif key[0] == 'noncompliant' and final_samples[key].endswith('3'):
        curr_sample = dict()
        curr_sample[('A', 0)] = 1
        curr_sample[('O', 0)] = prev_obs
        curr_sample[('R', 0)] = 0
        curr_sample[('A', 1)] = 1
        curr_sample[('O', 1)] = obs
        curr_sample[('R', 1)] = 1
        out_samples.append(curr_sample)

# LIST FINAL SAMPLES
# for sample in out_samples:
#     print sample

# SAVE SAMPLES
data = pd.DataFrame(out_samples)
data.to_csv('../data/qval_data.csv')

