from matplotlib import pyplot as plt
import numpy as np
import pickle

classifier = 'keras'
# classifier = 'pytorch'

# Data load
with open('temp/' + classifier + '/y1.pickle', 'rb') as rb:
    acc = pickle.load(rb)
with open('temp/' + classifier + '/y2.pickle', 'rb') as rb:
    loss = pickle.load(rb)

# Mean and std calculation
mean_y1 = np.mean(acc, axis=0)
mean_y2 = np.mean(loss, axis=0)

std_y1 = np.std(acc, axis=0)
std_y2 = np.std(loss, axis=0)

# Creating plot with loss
fig, ax1 = plt.subplots()
epochs = np.linspace(0, len(mean_y1))
plt.style.use('seaborn-whitegrid')

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.errorbar(epochs, mean_y2, yerr=std_y2, fmt=':', color=color, capsize=2)
ax1.fill_between(epochs, mean_y2 - std_y2, mean_y2 + std_y2, color=color, alpha=0.1)
ax1.tick_params(axis='y', labelcolor=color)

# Adding Twin Axes to plot using accuracy
ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Accuracy', color=color)
ax2.errorbar(epochs, mean_y1, yerr=std_y1, fmt=':', color=color, capsize=2)
ax2.fill_between(epochs, mean_y1 - std_y1, mean_y1 + std_y1, color=color, alpha=0.1)

ax2.tick_params(axis='y', labelcolor=color)

# Adding title
plt.title('Training process')

# Save fig
plt.savefig(classifier + ' training proces.svg')

# Show plot
plt.show()
