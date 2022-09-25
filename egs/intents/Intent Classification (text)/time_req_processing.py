import numpy as np

load_times = []
inference_times = []
with open('temp/times_laptop_keras.txt', 'r', encoding='utf-8-sig') as r:
    for line in r:
        vals = line.strip('\n').split(' ')
        load_times.append(float(vals[2]))
        inference_times.append(float(vals[-1]))

load_times_means = np.mean(load_times)
inference_times_means = np.mean(inference_times)

load_times_std = np.std(load_times)
inference_times_std = np.std(inference_times)

print('Load time: {} +- {}\nInference time: {} +- {}'.format(load_times_means, load_times_std, inference_times_means,
                                                          inference_times_std))
