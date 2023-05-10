import json

import numpy as np

with open('temp/experiments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
best_of = {}
with open('temp/bests.json', 'r', encoding='utf-8') as f:
    file = json.load(f)
    for mod in file:
        score = sum(file[mod]['accuracy'])
        if score == 10.0:
            best_of[mod] = file[mod]
print(len(best_of))

# some models have same architecture, but different seed -> remove them\
to_del = []
visited = []
for mod in list(best_of.keys()):
    for mod2 in list(best_of.keys()):
        if mod != mod2 and \
                mod.split('\t')[1] == mod2.split('\t')[1] and \
                mod.split('\t')[2] == mod2.split('\t')[2] and \
                mod.split('\t')[3] == mod2.split('\t')[3] and \
                mod.split('\t')[4] == mod2.split('\t')[4] and \
                mod.split('\t')[6] == mod2.split('\t')[6] and \
                visited.count(mod2) == 0:
            visited.append(mod2)
            to_del.append(mod2)
    visited.append(mod)

for mod in to_del:
    if mod in best_of:
        best_of.pop(mod)

print(len(best_of))

# find 10 best models with minimum time
best_of_best = {k: v for k, v in sorted(best_of.items(), key=lambda item: sum(item[1]['times']))}

best_of_best = dict(list(best_of_best.items())[:10])
print('Best time: ', sum(best_of_best[list(best_of_best.keys())[0]]['times']))
for best_model in best_of_best:
    print(best_model.split('\t')[0].replace('models/', ''))
    print('\tAccuracy: ', best_of_best[best_model]['accuracy'])
    print('\tTimes: ', best_of_best[best_model]['times'])
    print('\tParams: ', data[best_model])
