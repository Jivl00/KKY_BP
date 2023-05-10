import numpy as np


def data_stats(data):
    """ Print data statistics
    """
    print('Number of samples:', len(data))
    print('Number of unique intents:', len(set([line.split('\t')[1] for line in data])))
    print('Number of unique words:', len(set([word for line in data for word in line.split('\t')[0].split(' ')])))

    print('Number of samples per intent:', end=' ')
    for intent in set([line.split('\t')[1] for line in data]):
        continue
    print('{}'.format(len([line for line in data if line.split('\t')[1] == intent])))

    print('Average number of words per sample:', end=' ')
    print(np.mean([len(line.split('\t')[0].split(' ')) for line in data]))


path = 'dataset-2'
with open(path + '/train-cs.tsv', 'r', encoding='utf-8-sig') as f:
    train = [line for line in f if line != '\n']

with open(path + '/test-cs.tsv', 'r', encoding='utf-8-sig') as f:
    test = [line for line in f if line != '\n']

with open(path + '/dev-cs.tsv', 'r', encoding='utf-8-sig') as f:
    valid = [line for line in f if line != '\n']


print('Train data:')
data_stats(train)
print('Test data:')
data_stats(test)
print('Valid data:')
data_stats(valid)
