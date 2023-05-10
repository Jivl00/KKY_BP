

with open('data2.txt', 'r', encoding='utf-8-sig') as f:
    data = f.readlines()

with open('data_robot_tex.txt', 'w', encoding='utf-8-sig') as f:
    for line in data:
        if line == '\n':
            # f.write('\midrule\n')
            continue
        line = line.strip()
        vals = line.split(' ')
        intent = vals[0]
        vals = vals[1:]
        f.write(intent + ' & ' + ' '.join(vals) + ' \\\\\n')
