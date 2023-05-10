import json

with open('exp.json', 'r', encoding='utf-8-sig') as f:
    exp = json.load(f)

with open('temp.json', 'r', encoding='utf-8-sig') as f:
    temp = json.load(f)


from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

labels = temp['labels']
print(labels)
COLORS = ['blue', 'violet', 'peru','orange', 'lime', 'green', 'cyan', 'blue']
lab2COL = {label:col for label, col in zip(labels, COLORS)}

fig, ax = plt.subplots(2, 1, sharex = True)
fontsize = 12
plt.rcParams.update({'font.size': fontsize})

# Taking the states at the end of each iteration
ts =temp['ts']
steps = temp['steps']

# == AX0: Acc
ax[0].plot(ts, [exp[str(t)]['acc'] for t in ts], 'ko-', linewidth='2')
labels = labels[::-1]

for label in labels:
    try:
        acc_ = []
        for t in ts:
            if label in exp[str(t)]['labels']:
                wr = len([w for w in exp[str(t)]['wrongs'] if w[1] == label])
                ok = len([o for o in exp[str(t)]['oks'] if o[1] == label])
                acc_.append(ok/(ok+wr))

        ax[0].plot(ts[-len(acc_):], acc_, '-', color=lab2COL[label])
    except ZeroDivisionError:
        print(f'W: label {label} not in test data.')

s = temp['samples']
ax[0].set_title(f'# Test \nsamples: {s}', y=0.65, x=1.15, fontsize=fontsize)
ax[0].set_ylim([0, 1.1])
ax[0].set_ylabel('Test accuracy', fontsize=fontsize)
ax[0].tick_params(axis='y', labelsize=fontsize)
ax[0].grid()

# == AX1: Samples
for label in labels:
    ns = [exp[str(t)]['pt'][label] for t in ts if label in exp[str(t)]['labels']]
    ax[1].plot(ts[-len(ns):], ns, '-', color=lab2COL[label])

#w = 0.2
#for t, step in zip(ts, steps):
#    if step % 1 == 0:
#        for li, label in enumerate(labels):
#            ax[1].bar(t-w*(len(labels)/2)+(w*li), exp[t]['pt'][label], width=w, color=lab2COL[label])

#ax[1].plot(steps, [net.data.pk]*len(steps), '--', color='darkgreen', linewidth='2')

ax[1].set_ylabel('# Train samples', fontsize=fontsize)
ax[1].tick_params(axis='y', labelsize=fontsize)
#ax[1].set_ylim(bottom=0, top=max(n_train_max_)+2)
ax[1].set_xlabel('Step', fontsize=fontsize)
ax[1].set_xticks([t for i, t in enumerate(ts) if i%3 == 0])
ax[1].set_xticklabels([s for i, s in enumerate(steps) if i%3 == 0], fontsize=fontsize)


labels = labels[::-1]
ax[1].legend(
    bbox_to_anchor=(1.02, 1.55),
    handles=[mpatches.Patch(color='black', label='[mean]')]+[mpatches.Patch(color=lab2COL[label], label=label) for label in labels])

ax[1].grid()
axt = ax[1].twiny()
axt.set_xlabel('t (Realtime) [s]')
axt.spines["bottom"].set_position(("axes", -5.5))
print(ts[-1])
ticklabels = [t for t in range(int(float(ts[-1]))) if t % 30 == 0]
axt.set_xticks(ticklabels)
axt.set_xticklabels(ticklabels)

# Move twinned axis ticks and label from top to bottom
axt.xaxis.set_ticks_position("bottom")
axt.xaxis.set_label_position("bottom")

# Offset the twin axis below the host
axt.spines["bottom"].set_position(("axes", -0.45))

ax[0].text(-50, 1.15, '# Classes', color='red')
# ax[0].text(-90, 1.15, '# Classes', color='red')
ax[0].axvline(x=0, color='red', linestyle='--')
ax[1].axvline(x=0, color='red', linestyle='--')
ax[0].text(0, 1.15, exp[str(ts[0])]['m'], color='red')
for tim1, t in enumerate(ts[1:]):
    if exp[str(t)]['m'] != exp[str(ts[tim1])]['m']:
        ax[0].axvline(x=t, color='red', linestyle='--')
        ax[1].axvline(x=t, color='red', linestyle='--')
        ax[0].text(t, 1.15, exp[str(t)]['m'], color='red')

plt.show()
fig.savefig('test2.pdf', bbox_inches='tight', dpi=350)