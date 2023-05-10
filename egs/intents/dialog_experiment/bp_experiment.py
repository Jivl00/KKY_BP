import json

import torch
import torch.nn as nn

from net_t2l import NetT2L, T2LModel


def load_pretrained(fn):
    return torch.load(fn)


def load_middleblock(fn):
    mod = torch.load(fn)
    mod.output_layer = nn.Identity()
    return mod


# net = NetT2L()
# net = NetT2L(middleblock=load_middleblock(fn='model-intents-middleblock'))
# net = NetT2L(pretrained=load_pretrained(fn='model-intents-pretrained'))
net = NetT2L(pretrained=load_pretrained(fn='model-intents-pretrained-m4'))


from time import time

def now():
    return round(time()-t0, 2)

def shot(step, phase, t=None, **dialog):
    print('D:',dialog)
    loss, acc, wrongs, oks = net.evaluate(ret_wrongs=True, ret_oks=True)
    exp[t if t == 0 else now()] = {
        'step': step,
        'phase': phase,
        'labels': net.data.labels,
        'm': net.data.m,
        'pt': {label:net.data.data['train']['labels'].count(label) for label in net.data.labels},
        'loss': loss,
        'acc': acc,
        'wrongs': wrongs,
        'oks': oks,
        'dialog': dialog
    }

print('== Training loop ==')
exp = {}
steps = 20
t0 = time()

shot(step=0, phase='init', t=0)

for step in range(1, steps+1):
    ans = ''
    correction = ''
    prompt = input('\n\n> ')
    if prompt == 'done':
        break

    shot(step=step, phase='prompted', prompt=prompt)

    ans = net.predict(prompt)

    shot(step=step, phase='answered', prompt=prompt, ans=ans)

    correction = input(f'>> Ans: {ans} / Correct class (blank if OK): ')

    shot(step=step, phase='feedbacked', prompt=prompt, ans=ans, correction=correction.upper())

    if correction:
        net.learn(sample=prompt, label=correction, verbose=False)
        shot(step=step, phase='retrained', prompt=prompt, ans=ans, correction=correction)
    else:
        if prompt not in net.data.data['train']['samples']:
            net.learn(sample=prompt, label=ans, verbose=False)
            shot(step=step, phase='tuned', prompt=prompt, ans=ans, correction=correction)
        else:
            shot(step=step, phase='passed', prompt=prompt, ans=ans, correction=correction)

    loss, acc = net.evaluate()
    print(f'== Acc: {acc}, Loss: {loss}')



# for t, val in exp.items():
#     print(f'== [t={round(t, 2)}] Acc = {val["acc"]}, OKs: ({len(val["oks"])})')
#     for ok in val['oks']:
#         print(ok)
#
# for w in exp[max(list(exp.keys()))]['wrongs']:
#     print(w)


with open('exp.json', 'w', encoding='utf-8') as f:
    json.dump(exp, f, indent=4)

temp = {}

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

labels = net.data.labels
temp['labels'] = labels
temp['samples'] = net.data.pk



print(labels)
COLORS = ['yellow', 'green', 'blue', 'purple', 'orange', 'maroon', 'cyan', 'brown', 'pink']
lab2COL = {label:col for label, col in zip(labels, COLORS)}

fig, ax = plt.subplots(2, 1, sharex = True)

# Taking the states at the end of each iteration
ts = sorted([key for key, val in exp.items() if val['step'] == 0 or val['phase'] in ('tuned', 'passed', 'retrained')])
temp['ts'] = ts
steps = [exp[t]['step'] for t in ts]
temp['steps'] = steps

# == AX0: Acc
ax[0].plot(ts, [exp[t]['acc'] for t in ts], 'ko-', linewidth='2')

for label in labels:
    try:
        acc_ = []
        for t in ts:
            if label in exp[t]['labels']:
                wr = len([w for w in exp[t]['wrongs'] if w[1] == label])
                ok = len([o for o in exp[t]['oks'] if o[1] == label])
                acc_.append(ok/(ok+wr))

        ax[0].plot(ts[-len(acc_):], acc_, '-', color=lab2COL[label])
    except ZeroDivisionError:
        print(f'W: label {label} not in test data.')

ax[0].set_title(f'# test \nsamples: {net.data.pk}', y=0.65, x=1.15)
ax[0].set_ylim([0, 1.1])
ax[0].set_ylabel('Test Accuracy')
ax[0].grid()

# == AX1: Samples
for label in labels:
    ns = [exp[t]['pt'][label] for t in ts if label in exp[t]['labels']]
    ax[1].plot(ts[-len(ns):], ns, '-', color=lab2COL[label])

#w = 0.2
#for t, step in zip(ts, steps):
#    if step % 1 == 0:
#        for li, label in enumerate(labels):
#            ax[1].bar(t-w*(len(labels)/2)+(w*li), exp[t]['pt'][label], width=w, color=lab2COL[label])

#ax[1].plot(steps, [net.data.pk]*len(steps), '--', color='darkgreen', linewidth='2')

ax[1].set_ylabel('# train samples')
#ax[1].set_ylim(bottom=0, top=max(n_train_max_)+2)
ax[1].set_xlabel('step')
ax[1].set_xticks([t for i, t in enumerate(ts) if i%3 == 0])
ax[1].set_xticklabels([s for i, s in enumerate(steps) if i%3 == 0])

ax[1].legend(
    bbox_to_anchor=(1.02, 1.55),
    handles=[mpatches.Patch(color='black', label='[mean]')]+[mpatches.Patch(color=lab2COL[label], label=label) for label in labels])

ax[1].grid()
axt = ax[1].twiny()
axt.set_xlabel('t (realtime) [s]')
axt.spines["bottom"].set_position(("axes", -5.5))
print(ts[-1])
print(int(ts[-1]))
ticklabels = [t for t in range(int(ts[-1])) if t % 30 == 0]
axt.set_xticks(ticklabels)
axt.set_xticklabels(ticklabels)

# Move twinned axis ticks and label from top to bottom
axt.xaxis.set_ticks_position("bottom")
axt.xaxis.set_label_position("bottom")

# Offset the twin axis below the host
axt.spines["bottom"].set_position(("axes", -0.45))

ax[0].text(-45, 1.15, '# classes', color='red')
ax[0].axvline(x=0, color='red', linestyle='--')
ax[1].axvline(x=0, color='red', linestyle='--')
ax[0].text(0, 1.15, exp[ts[0]]['m'], color='red')
for tim1, t in enumerate(ts[1:]):
    if exp[t]['m'] != exp[ts[tim1]]['m']:
        ax[0].axvline(x=t, color='red', linestyle='--')
        ax[1].axvline(x=t, color='red', linestyle='--')
        ax[0].text(t, 1.15, exp[t]['m'], color='red')

plt.show()
fig.savefig('test.pdf', bbox_inches='tight', dpi=350)

with open('temp.json', 'w', encoding='utf-8') as f:
    json.dump(temp, f, indent=4)