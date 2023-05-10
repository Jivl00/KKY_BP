import numpy as np
from sentence_transformers import SentenceTransformer

# path = 'dataset-2'
path = 'data_robot'
bert_path = 'fav-kky/FERNET-C5'
model = SentenceTransformer(bert_path)
bert_model = model


def encode_data(file_name):
    new_file = []
    np.set_printoptions(linewidth=np.inf)
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        file = [line for line in f]
    for line in file:
        if line == '\n':
            new_file.append(line)
            continue
        vals = line.split('\t')
        encoded = bert_model.encode(vals[0])
        vals[1] = vals[1].strip('\n')
        new_file.append(vals[0] + '\t' + vals[1] + '\t' + str(encoded) + '\n')
    with open(file_name, 'w', encoding='utf-8-sig') as f:
        for line in new_file:
            f.write(line)


# encode_data(path + '/train-cs.tsv')
# encode_data(path + '/test-cs.tsv')
# encode_data(path + '/dev-cs.tsv')
encode_data(path + '/data_robot_shuffled.txt')
# encode_data(path + '/data_robot_bigger.txt')