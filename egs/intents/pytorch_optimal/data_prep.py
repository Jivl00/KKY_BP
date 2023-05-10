import os
import re
from logs import log
import pickle
from sentence_transformers import SentenceTransformer
from transformers import logging

logging.set_verbosity_error()  # shut up Bert warning

# Load czech BERT model
bert_path = 'fav-kky/FERNET-C5'
bert_model = SentenceTransformer(bert_path)
if not os.path.exists(bert_path):
    bert_model.save(bert_path)


def read_data(data_file):
    """ Read data
    returns: samples, targets (dicts keyed by a sample index)
    """

    log('Loading training data.')
    samples = []
    targets = []

    # Read data (encoding='utf-8-sig' => omit BOM)
    with open(data_file, 'r', encoding='utf-8-sig') as r:
        # Read samples and targets
        for ind, line in enumerate(r):
            line = line.lower()
            line = clean_text(line).strip('\n')  # remove punctuation and '\n'
            line = clean_numbers(line)  # replace numbers with '#'
            vals = line.split(' ')
            sentence = line.replace(vals[0] + ' ', '', 1)

            # Feature vector representing a sample
            samples.append(bert_model.encode(sentence))

            # Corresponding target
            targets.append(vals[0])

    target_names = set(targets)
    target_names = sorted(target_names)

    # Store data (serialize)
    with open('temp/target_names.pickle', 'wb') as wb:
        pickle.dump(target_names, wb, protocol=pickle.HIGHEST_PROTOCOL)
    log('Found {} intent classes.'.format(len(target_names)))

    return samples, targets, target_names


def read_sentence(line):
    line = line.lower()
    line = clean_text(line).strip('\n')  # remove punctuation and '\n'
    line = clean_numbers(line)  # replace numbers with '#'

    # Feature vector representing a sample
    return bert_model.encode(line)


def get_target_names():
    # Load data (deserialize)
    with open('temp/target_names.pickle', 'rb') as rb:
        unserialized_data = pickle.load(rb)
    return unserialized_data


def clean_numbers(x):
    x = re.sub('([0-9])', '#', x)
    return x


def clean_text(x):
    text = re.sub(r'[^\w\s]', '', x)
    return text
