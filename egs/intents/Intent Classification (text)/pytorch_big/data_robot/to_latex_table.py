

with open('data_robot.txt', 'r', encoding='utf-8-sig') as f:
    data = f.readlines()

with open('data_robot_tex.txt', 'w', encoding='utf-8-sig') as f:
    for line in data:
        if line == '\n':
            f.write('\midrule')
            continue
        line = line.strip()
        vals = line.split('\t')
        f.write(vals[1].upper() + ' & ' + vals[0] + '\\\\ \n')
