import json
import numpy as np
import matplotlib.pyplot as plt


with open('data_robot/epochs_time2.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

times = []
accuracies = []
labels = ['']
for epoch in data:
    labels.append(epoch)
    accuracies.append(data[epoch]['accuracies'])
    times.append(data[epoch]['times'])

# plot heatmap for accuracy and heatmap for time
font_size = 12
fig, ax1 = plt.subplots(2, 1, figsize=(9, 7))
plt.rcParams.update({'font.size': font_size})

# accuracy heatmap
ax1[0].tick_params(axis='both', labelsize=font_size+2)
ax1[0].set_xticks(np.arange(len(accuracies[0])))
ax1[0].set_yticklabels(labels)
ax1[0].set_xlabel('Steps', fontsize=font_size+2)
im = ax1[0].imshow(accuracies, cmap='YlGn', interpolation='nearest', aspect=0.9, vmin=0.5, vmax=1.1)

# show all accuracy values in the heatmap
for i in range(len(accuracies)):
    for j in range(len(accuracies[0])):
        ax1[0].text(j, i, round(accuracies[i][j], 3), va='center', ha='center', fontsize=font_size)

# time heatmap
im = ax1[1].imshow(times, cmap='winter', interpolation='nearest', aspect=0.9, vmin=0.008, vmax=0.5)
ax1[1].tick_params(axis='both', which='major', labelsize=font_size+2)
ax1[1].set_xticks(np.arange(len(times[0])))
ax1[1].set_yticklabels(labels)
ax1[1].set_xlabel('Steps', fontsize=font_size+2)


# show all accuracy values in the heatmap
for i in range(len(times)):
    for j in range(len(times[0])):
        ax1[1].text(j, i, round(times[i][j], 3), va='center', ha='center', fontsize=font_size)

plt.tight_layout()
plt.savefig('data_robot/heatmap_dialog2.pdf', bbox_inches='tight')
plt.show()



