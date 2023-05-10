import matplotlib.pyplot as plt
import numpy as np

with open('data_robot/data_robot_bigger.txt', 'r', encoding='utf-8-sig') as f:
    data = f.readlines()

with open('data_robot/acc.txt', 'r', encoding='utf-8-sig') as f:
    accu = f.readlines()

accuracies = []
for acc in accu:
    accuracies.append(float(acc.strip()))

classes_counts = []
classes = set()
samples = []
samples_counter = 0
classes_with_num_samples = {}
mean_classes = [] # mean value of samples per classes
std_classes = []

for line in data[:2]:
    if line == '\n':
        continue
    class_name = line.split('\t')[1].strip()
    classes.add(class_name)
    try :
        classes_with_num_samples[class_name] += 1
    except KeyError:
        classes_with_num_samples[class_name] = 1
    samples_counter += 1
for line in data[2:]:
    if line == '\n':
        classes_counts.append(len(classes))
        samples.append(samples_counter)
        mean_classes.append(np.mean(list(classes_with_num_samples.values())))
        std_classes.append(np.std(list(classes_with_num_samples.values())))
        continue

    class_name = line.split('\t')[1].strip()
    classes.add(class_name)
    try :
        classes_with_num_samples[class_name] += 1
    except KeyError:
        classes_with_num_samples[class_name] = 1
    samples_counter += 1


discrete_time = [i for i in range(len(classes_counts))]
print(classes_counts)
print(samples)
print(accuracies)
print(discrete_time)
print(classes_with_num_samples)
print(mean_classes)
print(std_classes)


font_size = 17
fig, ax1 = plt.subplots(2, 1, figsize=(10, 8))
plt.rcParams.update({'font.size': font_size})
ax1[0].step(discrete_time, classes_counts, where='post', color='tab:red')
ax1[0].set_xlabel('Steps', fontsize=font_size)
ax1[0].set_ylabel('Classes', color='tab:red', fontsize=font_size)
ax1[0].tick_params(axis='y', labelcolor='tab:red', labelsize=font_size)
ax1[0].tick_params(axis='x', labelsize=font_size)

ax2 = ax1[0].twinx()
ax2.bar(discrete_time, samples, width=0.5, align='center', alpha=0.5)
ax2.set_ylabel('Samples', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')


ax3 = ax1[0].twinx()

ax3.plot(discrete_time, mean_classes, color='tab:green')
ax3.fill_between(discrete_time, np.array(mean_classes) - np.array(std_classes), np.array(mean_classes) + np.array(std_classes), alpha=0.5, color='tab:green')
ax3.set_ylabel('Mean samples per class', color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')
ax3.spines['right'].set_position(('axes', 1.125))
plt.grid()

# heatmap of accuracies
ax1[1].imshow(np.array(accuracies).reshape(1, -1), cmap='YlGn', interpolation='nearest', aspect=0.5, vmin=0.9, vmax=1.1)
# show all accuracy values in the heatmap
for i, acc in enumerate(accuracies):
    ax1[1].text(i, 0, round(acc, 2), va='center', ha='center', fontsize=font_size)
ax1[1].set_xlabel('Accuracy', fontsize=font_size)
ax1[1].tick_params(axis='x', labelsize=font_size)
ax1[1].set_yticks([])



fig.tight_layout()
plt.subplots_adjust(hspace=-0.30)

plt.savefig('dialog_training_graph2.pdf', bbox_inches='tight')
plt.show()