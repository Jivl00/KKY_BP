# print error bars for the best model of the training process
import numpy

with open('pytorch_optimal_train_process.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    acc = []
    loss = []
    for line in lines:
        vals = line.split('\t')
        vals[1] = vals[1].replace('[', '').replace(']', '')
        vals[2] = vals[2].replace('[', '').replace(']', '')
        acc.append(numpy.fromstring(vals[1], dtype=float, sep=','))
        loss.append(numpy.fromstring(vals[2], dtype=float, sep=','))

acc = numpy.array(acc)
loss = numpy.array(loss)

# error bars
acc_mean = numpy.mean(acc, axis=0)
acc_std = numpy.std(acc, axis=0)
loss_mean = numpy.mean(loss, axis=0)
loss_std = numpy.std(loss, axis=0)

# plot
import matplotlib.pyplot as plt

# Creating plot with loss
fig, ax1 = plt.subplots()

plt.rcParams.update({'font.size': 14})

color = 'tab:red'
ax1.tick_params(axis='both', labelsize=14)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Loss', color=color, fontsize=14)
ax1.plot(loss_mean, color=color)
ax1.fill_between(numpy.arange(len(loss_mean)), loss_mean - loss_std, loss_mean + loss_std, alpha=0.2, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Adding Twin Axes to plot using accuracy
ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(acc_mean, color=color)
ax2.fill_between(numpy.arange(len(acc_mean)), acc_mean - acc_std, acc_mean + acc_std, alpha=0.2, color=color)

ax2.tick_params(axis='y', labelcolor=color)

# Adding title
plt.grid()

# Show plot
plt.savefig('pytorch_optimal_train_process.pdf')
plt.show()