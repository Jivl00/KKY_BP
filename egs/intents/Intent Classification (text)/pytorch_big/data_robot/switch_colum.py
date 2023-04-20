with open('data.txt', 'r', encoding='utf-8-sig') as f:
    data = f.readlines()

with open('data.txt', 'w', encoding='utf-8-sig') as f:
    for line in data:
        line = line.strip()
        vals = line.split('\t')
        f.write(vals[1] + '\t' + vals[0] + '\n')