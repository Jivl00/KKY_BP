import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # shut up tensorflow debug messages

# Force use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datetime import datetime
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.client import device_lib
import numpy as np


def fix_ts():
    """ ts format: YYYY-MM-DDThh:mm:ss """
    
    return datetime.now().isoformat().split('.')[0]


def log(s):
    """ Print a LOG message """

    print('[LOG '+fix_ts()+'] '+s)


def line2list(line):
    return line.strip().split(',')


def get_limits(feature, samples):
    """ Given a feature, get the minimal and maximal values """

    vals_this_feature = [sample[feature] for sample in samples.values()]

    return np.min(vals_this_feature), np.max(vals_this_feature)


def norm(val, low, high, new_low=0, new_high=1):
    """ Linearly scale a value from range <low, high> into 
        a new range <new_low, new_high>
    """

    return (new_high-new_low)/(high-low)*(val-low)+new_low


def read_data(data_file):
    """ Read and normalize data
    returns: samples, targets (dicts keyed by a sample index)
    sample: dict keyed by a feature name
    target: binary value of {0, 1}
    """

    samples = {}
    targets = {}

    # Read data (encoding='utf-8-sig' => omit BOM)
    with open(data_file, 'r', encoding='utf-8-sig') as cfr:
        
        # Read feature names (CSV header), last col ~ target
        features = line2list(cfr.readline())[:-1]

        log('Found {} features.'.format(len(features)))
        
        # Read samples and targets, add them to dicts (key ~ index)
        for ind, line in enumerate(cfr.readlines()):
            vals = line2list(line)

            # Feature vector representing a sample
            samples[ind] = {feature:float(val) for feature, val in zip(features, vals[:-1])}

            # Corresponding target
            targets[ind] = float(vals[-1])

    # Get limits and normalize samples
    log('Data loaded. Normalizing...')
    for feature in features:
        low, high = get_limits(feature, samples)
        for sample in samples.values():
            sample[feature] = norm(sample[feature], low, high)

    log('Data normalized.')

    return samples, targets, sorted(features)


def show_data_stats(samples, targets, print_data=False):
    """ Print out some data stats """

    log('Showing data stats...')
    print('Number of samples: {}'.format(len(samples)))
    print('Number of positive samples: {}'.format(sum([1 for t in targets.values() if t == 1.0])))
    print('Number of negative samples: {}'.format(sum([1 for t in targets.values() if t == 0.0])))

    if print_data:
        for ind in samples.keys():
            print('{}\t{}\t{}'.format(ind, targets[ind], [(f, round(val, 2)) for f, val in samples[ind].items()]))


def split_data(targets, split_ratio=(0.8, 0.1, 0.1), seed=np.random.randint(0, 100)):
    """ Split data to disjunctive sets: train, dev, test
        return: dict
            keys: 'train', 'dev', 'test
            values: list of samples indices
    """

    split = {'train': [], 'dev': [], 'test': []}

    # Set random seed for shuffling this split
    np.random.seed(seed)

    ## Carefully and equally distribute positive and negative samples
    negs = [ind for ind, val in targets.items() if val == 0.0]
    poss = [ind for ind, val in targets.items() if val == 1.0]

    for kind in (negs, poss):
        np.random.shuffle(kind)
        for i, ind in enumerate(kind):
            if i/len(kind) < split_ratio[0]:
                split['train'].append(ind)
            elif split_ratio[0] < i/len(kind) < split_ratio[0]+split_ratio[1]:
                split['dev'].append(ind)
            else:
                split['test'].append(ind)
        
    log('Got {} train, {} dev and {} test samples.'.format(len(split['train']), len(split['dev']), len(split['test'])))

    return split


def form_data(samples, targets, split, feats):
    """ Prepare matrices for machine learning in Keras
        return dict with keys: 'x_train', 'y_train', 'x_dev', 'y_dev', 'x_test', 'y_test'
    """

    data = {}
    for group, inds in split.items():
        data['x_'+group] = np.array([[samples[ind][f] for f in feats] for ind in inds])
        data['y_'+group] = np.array([targets[ind] for ind in inds], ndmin=2).T
        log('Group {}, X shape: {}, Y: shape: {}'.format(group, data['x_'+group].shape, data['y_'+group].shape))

    return data


def design_model(inp_shape, out_units, params, print_summary=True):
    log('Building the model...')
    
    # Input layer
    NET_I = Input(shape=inp_shape)

    # Hidden layers
    NET = NET_I
    for n_cells in params['hidden_layers']:
        NET = Dense(units=n_cells, activation='sigmoid')(NET)

    # Output layer
    NET_O = Dense(units=out_units, activation='sigmoid')(NET)

    model = Model(inputs=NET_I, outputs=NET_O)

    if print_summary:
        model.summary()

    return model


if __name__ == '__main__':

    log('TF loaded. Using '+('GPU' if 'GPU' in str(device_lib.list_local_devices()) else 'CPU'))

    # -- SETTINGS
    DATA_FILE = '../data/heart.csv'                 # source CSV file
    DATA_SPLIT_RATIO = (0.8, 0.1, 0.1)              # (train, dev, test)

    NET_PARAMS = {
        'hidden_layers': [15, 10],
        'learning_rate': 0.0005,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy'],
        'epochs': 500,
        'batch_size': 16,
        'do_fit': True,
        'overwrite_best_model': True
    }


    ## -- DATA PREPROCESSING

    # Read and normalize data, sort features
    samples, targets, feats = read_data(data_file=DATA_FILE)

    # Get basic data stats (you can print all samples to check)
    show_data_stats(samples, targets, print_data=False)

    # Split data (train, dev, test)
    split = split_data(targets, DATA_SPLIT_RATIO)

    # Form data for Keras
    data = form_data(samples, targets, split, feats)


    ## -- NEURAL NETWORK DESIGN

    # Network architecture
    model = design_model(inp_shape=data['x_train'][0].shape, out_units=1, params=NET_PARAMS)

    # Optimizer
    optimizer = Adam(learning_rate=NET_PARAMS['learning_rate'])

    # Network compilation
    model.compile(loss=NET_PARAMS['loss'], optimizer=optimizer, metrics=NET_PARAMS['metrics'])
    log('Model compiled.')


    ## -- NEURAL NETWORK TRAINING

    log('Training the model...')

    # Save the best gained model to this path
    if NET_PARAMS['overwrite_best_model']:
        best_model_path = '../models/best.h5'
    else:
        best_model_path = '../models/model_{}.h5'.format(fix_ts().replace(':', '-'))

    # Train the network
    if NET_PARAMS['do_fit']:
        model.fit(data['x_train'], data['y_train'], 
                validation_data=(data['x_dev'], data['y_dev']),
                epochs=NET_PARAMS['epochs'], 
                batch_size=NET_PARAMS['batch_size'], 
                shuffle=True, 
                verbose=True, 
                callbacks=[ModelCheckpoint(best_model_path, monitor='val_loss', verbose=False, save_best_only=True, save_weights_only=False)])
    
    
    ## -- NEURAL NETWORK EVALUATION

    # Load model for evaluation
    evaluated_model = best_model_path
    #evaluated_model = 'path_to_your_model_for_evaluation'

    model = load_model(evaluated_model)

    # Evaluate on test data
    y_pred = model.predict(data['x_test'])

    test_loss, test_acc = model.evaluate(data['x_test'], data['y_test'])
    print(test_loss, test_acc)
    
    for yi, ui in zip(y_pred, data['y_test']):
        print(yi[0], ui[0])

    