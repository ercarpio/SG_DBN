import os
import math
import numpy

strings = ['.tfrecord', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_', '-']
categories = dict()
file_lists = dict()

for root, subFolders, files in os.walk('/home/assistive-robotics/ITBN_tfrecords/'):
    for f in files:
        original_f = f
        for str in strings:
            f = f.replace(str, '')
        count = categories.get(f, 0)
        categories[f] = count + 1
        list = file_lists.get(f, [])
        list.append(os.path.join(root, original_f))
        file_lists[f] = list

for cat in categories.keys():
    categories[cat] = int(math.ceil(categories[cat] * 0.25))

for cat in file_lists:
    print (file_lists[cat])

for cat in file_lists:
    files = numpy.random.choice(numpy.array(file_lists[cat]), categories[cat], replace=False)
    # for f in files:
        # os.rename(f, f.replace('.tfrecord', '_validation.tfrecord'))

categories = dict()
for root, subFolders, files in os.walk('/home/assistive-robotics/ITBN_tfrecords/'):
    for f in files:
        original_f = f
        for str in strings:
            f = f.replace(str, '')
        count = categories.get(f, 0)
        categories[f] = count + 1

for cat in categories:
    print (cat)
    print (categories[cat])