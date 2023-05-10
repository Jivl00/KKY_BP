import pickle

from transformers import logging
import os
import random
import numpy as np
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up tensorflow debug messages
import tensorflow as tf
import keras.utils.np_utils
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

tf.random.set_seed(0)
random.seed(0)

bert_path = 'fav-kky/FERNET-C5'
bert_model = SentenceTransformer(bert_path)

# Force use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.set_verbosity_error()  # shut up Bert warning

NET_PARAMS = {
    'hidden_layers': [512, 256],
    'learning_rate': 0.003,
    'learning_rate_decay': 0.9,
    'batch_size': 2048,
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'epochs': 300
}


def form(file):
    samples = []
    targets = []
    for line in file:
        if line == '\n':
            continue
        vals = line.split('\t')
        sample = vals[2].strip('\n').replace('[', '').replace(']', '').split(' ')
        sample = list(filter(None, sample))
        sample = np.array(list(np.float_(sample)))
        samples.append(sample)

        targets.append(vals[1])
    target_names = set(targets)

    x = np.array(samples)
    samples = x

    target_numbers = []
    for target in targets:
        target_numbers.append(list(target_names).index(target))
    targets = keras.utils.to_categorical(np.array(target_numbers), num_classes=len(target_names))
    print('Data formed')

    return samples, targets, target_names


def read_data(train, test, valid):
    """ Read data
    returns: samples, targets (dicts keyed by a sample index)
    """

    print('Loading data.')
    samples = {'train': [], 'test': [], 'valid': []}
    targets = {'train': [], 'test': [], 'valid': []}

    samples['train'], targets['train'], target_names = form(train)
    samples['test'], targets['test'], _ = form(test)
    samples['valid'], targets['valid'], target_names = form(valid)
    print('Found {} intent classes.'.format(len(target_names)))
    #
    return samples, targets, target_names


def get_target_names():
    # Load data (deserialize)
    with open('models/target_names.pickle', 'rb') as rb:
        unserialized_data = pickle.load(rb)
    return list(unserialized_data)


def design_model(inp_shape, out_units, params, print_summary=True):
    print('Building the model...')

    # Input layer
    NET_I = Input(shape=inp_shape)

    # Hidden layers
    NET = NET_I
    for n_cells in params['hidden_layers']:
        NET = Dense(units=n_cells, activation='relu')(NET)

    # Output layer
    NET_O = Dense(units=out_units, activation='softmax')(NET)

    model = Model(inputs=NET_I, outputs=NET_O)

    if print_summary:
        model.summary()

    return model


def evaluate_model(best_model_path, x_test, y_test, target_names, print_detail):
    # Load model for evaluation
    model = load_best_model(best_model_path=best_model_path)

    # Evaluate on test data
    y_pred = model.predict(x_test)

    y_test_num = list(np.argmax(y_test, axis=1))
    y_pred_num = list(np.argmax(y_pred, axis=1))
    print("Model accuracy: {}".format(accuracy_score(y_test_num, y_pred_num)))

    if print_detail:  # Print predicted vs test classes
        print("Predicted vs test classes:")

def read_sentence(line):
    line = line.lower()
    line = line.strip('\n')  # remove punctuation and '\n'

    # Feature vector representing a sample
    return bert_model.encode(line)


def predict_single(user_input, model):
    """
    Predict class for given input

    """
    target_names = get_target_names()

    # Form the text for classification
    user_input = read_sentence(user_input)
    user_input = np.array([user_input])

    # Make prediction
    pred = model.predict(user_input, verbose=0)
    pred = np.argmax(pred)

    return target_names[pred]


def load_best_model(best_model_path="models/best.h5"):
    return load_model(best_model_path)


def train_process_graph(accu, losses):
    # Creating plot with loss
    fig, ax1 = plt.subplots()

    plt.rcParams.update({'font.size': 14})

    color = 'tab:red'
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel('Epochs', fontsize=14)
    ax1.set_ylabel('Loss', color=color, fontsize=14)
    ax1.plot(losses, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding Twin Axes to plot using accuracy
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accu, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding title
    plt.title('Training process')
    plt.grid()

    # Show plot
    plt.savefig('temp/train_process_keras.pdf')
    plt.show()


if __name__ == '__main__':
    # Load data
    path = 'dataset-2'
    with open(path + '/train-cs.tsv', 'r', encoding='utf-8-sig') as f:
        train = [line for line in f]
    with open(path + '/test-cs.tsv', 'r', encoding='utf-8-sig') as f:
        test = [line for line in f]
    with open(path + '/dev-cs.tsv', 'r', encoding='utf-8-sig') as f:
        valid = [line for line in f]

    samples, targets, target_names = read_data(train, test, valid)

    # -- SETTINGS

    # -- NEURAL NETWORK DESIGN

    # Network architecture
    neurons_num = {'inp_shape': len(samples['valid'][0]), 'out_shape': len(target_names)}
    model = design_model(inp_shape=len(samples['valid'][0]), out_units=len(target_names), params=NET_PARAMS)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=NET_PARAMS['learning_rate'])

    # Network compilation
    model.compile(loss=NET_PARAMS['loss'], optimizer=optimizer, metrics=NET_PARAMS['metrics'])
    print('Model compiled.')

    # -- NEURAL NETWORK TRAINING

    # Save the best gained model to this path
    best_model_path = "models/best.h5"

    # Train the network
    if True:
        print('Training the model...')
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=NET_PARAMS['epochs']/5, verbose=0)
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=NET_PARAMS['epochs']/50, verbose=0)
        h = model.fit(samples['train'], targets['train'],
                      epochs=NET_PARAMS['epochs'],
                      batch_size=NET_PARAMS['batch_size'],
                      shuffle=False,
                      verbose=0,
                      validation_data=(samples['valid'], targets['valid']),
                      callbacks=[
                          ModelCheckpoint(best_model_path, monitor='val_loss', verbose=False, save_best_only=False,
                                          save_weights_only=False)])
        print('Model trained.')
        train_process_graph(h.history['accuracy'], h.history['loss'])

    # -- NEURAL NETWORK EVALUATION

    # Evaluate on test data (test data = train data - insufficient amount of data)
    evaluate_model(best_model_path, samples['test'], targets['test'], target_names, print_detail=True)

