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
                recordings[line] = lines[i + 1]
            elif recordings[line] != lines[i + 1]:
                count_repeated += 1
                recordings[line + 'r'] = lines[i + 1]
            else:
                count_discarded += 1
    print 'Data loaded.\nTotal:\t\t{0}\nRepeated:\t{1}\nDiscarded:\t{2}\n'.format(
        len(recordings), count_repeated, count_discarded)

samples = dict()

for key in sorted(recordings.keys()):
    keys = key.split('_')
    if not keys[3].endswith('r'):
        samples[(keys[0], int(keys[1]), int(keys[2]), keys[3])] = recordings[key]
    else:
        samples[(keys[0], int(keys[1]), -int(keys[2]), keys[3].replace('r',''))] = recordings[key]

for key in sorted(samples.keys()):
    print key
