import os

validation_set = list()
training_set = list()
for root, dirs, files in os.walk("../../../ITBN_tfrecords"):
    for f in files:
        if 'validation' in f:
            validation_set.append(os.path.join(root, f))
        else:
            training_set.append(os.path.join(root, f))

# to_remove = ['../../../ITBN_tfrecords/test_',
#              '.tfrecord', '/', '_', 'validation', '-']

# for i in range(0, 10):
#     to_remove.append(str(i))

training_type = dict()
validation_type = dict()

print("TRAINING")
for example in training_set:
    # for str in to_remove:
    #     example = example.replace(str, '')
    example = example[len('../../../ITBN_tfrecords/test_10/'):].replace('.tfrecord', '')

    print(example)
    if training_type.get(example, 0) == 0:
        training_type[example] = 1
    else:
        training_type[example] = training_type[example] + 1

print("\nVALIDATION")
for example in validation_set:
    # for str in to_remove:
    #     example = example.replace(str, '')
    example = example[len('../../../ITBN_tfrecords/test_10/'):].replace('.tfrecord', '').replace('_validation', '')
    print(example)
    if validation_type.get(example, 0) == 0:
        validation_type[example] = 1
    else:
        validation_type[example] = validation_type[example] + 1

print(training_type)
print(validation_type)

print(len(training_set))
print(len(validation_set))