import os

validation_set = list()
training_set = list()
for root, dirs, files in os.walk("../ITBN_tfrecords"):
    for f in files:
        if 'validation' in f:
            validation_set.append(os.path.join(root, f))
        else:
            training_set.append(os.path.join(root, f))

print("TRAINING")
for example in training_set:
    print(example)

print("VALIDATION")
for example in validation_set:
    print(example)
