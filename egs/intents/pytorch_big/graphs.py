import matplotlib.pyplot as plt
import json

with open('data_for_graphs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

data_60 = data['130']
for key in data_60:
    data_60[key] = data_60[key][:120]


# plt.plot(data_60['train_losses_ft'])
# plt.plot(data_60['train_losses_scratch'])
# plt.plot(data_60['train_losses_tl'])
# plt.title('Loss')
# plt.grid()
# plt.legend(['Fine-tuning', 'Scratch', 'Transfer Learning'])
# plt.show()

fig, ax1 = plt.subplots()

plt.rcParams.update({'font.size': 14})

color = 'tab:red'
ax1.tick_params(axis='both', labelsize=14)
ax1.set_xlabel('Epochs', fontsize=14)
ax1.set_ylabel('Loss', color=color, fontsize=14)
ax1.plot(data_60['train_losses_scratch'], color=color)
ax1.plot(data_60['train_losses_ft'], color=color, linestyle='--')
ax1.plot(data_60['train_losses_tl'], color=color, linestyle=':')
ax1.tick_params(axis='y', labelcolor=color)

# Adding Twin Axes to plot using accuracy
ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(data_60['train_accu_ft'], color=color, linestyle='--')
ax2.plot(data_60['train_accu_scratch'], color=color)
ax2.plot(data_60['train_accu_tl'], color=color, linestyle=':')
ax2.tick_params(axis='y', labelcolor=color)

# Add legend
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=14)
# place legend center of plot
# black color for legend
color = 'tab:gray'
ax2.plot([], [], label="Model 1", color=color)
ax2.plot([], [], label="Model 2", color=color, linestyle='--')
ax2.plot([], [], label="Model 3", color=color, linestyle=':')

fig.legend(loc='center left', bbox_to_anchor=(0.65, 0.5), ncol=1, fontsize=12)
# Adding title
plt.grid()

# Show plot
plt.savefig('train_process_130.pdf', bbox_inches='tight')
plt.show()

# validation losses
color = 'tab:red'
plt.plot(data_60['dev_losses_scratch'], color=color)
plt.plot(data_60['dev_losses_ft'], color=color, linestyle='--')
plt.plot(data_60['dev_losses_tl'], color=color, linestyle=':')
plt.tick_params(axis='both', labelcolor=color)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', color=color, fontsize=14)
plt.grid()
plt.legend(['Model 1', 'Model 2', 'Model 3'])
plt.savefig('dev_loss_130.pdf', bbox_inches='tight')
plt.show()
