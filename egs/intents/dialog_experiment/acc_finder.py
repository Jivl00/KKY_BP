
import json

with open('exp.json', 'r', encoding='utf-8-sig') as f:
    exp = json.load(f)

with open('data_robot_tex.txt', 'r', encoding='utf-8-sig') as f:
    data = f.readlines()

accs = []
for i,key in enumerate(exp):
    if exp[key]["phase"] in ('tuned', 'passed', 'retrained'):
        acc = exp[key]["acc"]
        acc = round(acc*100, 2)
        accs.append("& \\makecell{" + str(acc) + " \\%}")

for i, line in enumerate(data):
    data[i] = line.strip().replace('\\\\', ' '.join(accs[i].split(' ')) + ' \\\\\n')
with open('data_robot_tex.txt', 'w', encoding='utf-8-sig') as f:
    for line in data:
        f.write(line)
