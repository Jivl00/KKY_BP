import numpy as np

with open('data.txt', 'r', encoding='utf-8-sig') as f:
    data = [line for line in f]

print('Data:')
print("Number of samples:", len(data))
print("Number of unique intents:", len(set([line.split('\t')[0] for line in data])))
print("Number of unique words:", len(set([word for line in data for word in line.split('\t')[1].split(' ')])))

print("Number of samples per intent:")
for intent in set([line.split('\t')[0] for line in data]):
    print(intent, ':', len([line for line in data if line.split('\t')[0] == intent]))

print("Average number of words per sample:")
print(np.mean([len(line.split('\t')[1].split(' ')) for line in data]))