import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # shut up tensorflow debug messages

import random
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import tensorflow as tf
import keras.utils.np_utils
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from sklearn.metrics import accuracy_score
from intent_classifier.data_prep import *
from intent_classifier.logs import log, fix_ts

tf.random.set_seed(0)
random.seed(0)


# Force use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def form_data(samples, targets, target_names):
    """ Prepare matrices for machine learning in Keras
        return dict with keys: 'x_train', 'y_train'
    """

    data = {'x_train': np.array(samples)}

    target_numbers = []
    for target in targets:
        target_numbers.append(target_names.index(target))
    data['y_train'] = keras.utils.to_categorical(np.array(target_numbers), num_classes=len(target_names))
    log('Data formed')

    return data, target_numbers


def design_model(inp_shape, out_units, params, print_summary=True):
    log('Building the model...')

    # Input layer
    NET_I = Input(shape=inp_shape)

    # Hidden layers
    NET = NET_I
    for n_cells in params['hidden_layers']:
        NET = Dense(units=n_cells, activation='relu')(NET)

    # Output layer
    NET_O = Dense(units=out_units, activation='sigmoid')(NET)

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
    skplt.metrics.plot_confusion_matrix(
        y_test_num, y_pred_num, x_tick_rotation=90)
    plt.show()

    if print_detail:  # Print predicted vs test classes
        y_pred_names = [target_names[i] for i in y_pred_num]
        y_test_names = [target_names[i] for i in y_test_num]

        for i in range(len(y_test_num)):
            print(
                'Predicted: {} {}   Test: {} {}'.format(y_pred_num[i], y_pred_names[i], y_test_num[i], y_test_names[i]))


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


def load_best_model(best_model_path="intent_classifier/models/best.h5"):
    return load_model(best_model_path)


if __name__ == '__main__':

    log('TF loaded. Using ' + ('GPU' if 'GPU' in str(device_lib.list_local_devices()) else 'CPU'))

    # -- DATA PREPROCESSING
    DATA_FILE = 'data.txt'  # source txt file

    # Read data and form samples vectors using BERT
    samples, targets, target_names = read_data(data_file=DATA_FILE)

    # Form data for Keras
    data, target_numbers = form_data(samples, targets, target_names)

    # -- SETTINGS

    NET_PARAMS = {
        'hidden_layers': [35],
        'learning_rate': 0.1,
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
        'epochs': 20,
        'batch_size': 10,
        'do_fit': True,
        'overwrite_best_model': True
    }

    # -- NEURAL NETWORK DESIGN

    # Network architecture
    model = design_model(inp_shape=len(data['x_train'][0]), out_units=len(target_names), params=NET_PARAMS)

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=NET_PARAMS['learning_rate'])

    # Network compilation
    model.compile(loss=NET_PARAMS['loss'], optimizer=optimizer, metrics=NET_PARAMS['metrics'])
    log('Model compiled.')

    # -- NEURAL NETWORK TRAINING

    # Save the best gained model to this path
    if NET_PARAMS['overwrite_best_model']:
        best_model_path = "intent_classifier/models/best.h5"
    else:
        best_model_path = "models/model_{}.h5".format(fix_ts().replace(':', '-'))

    # Train the network
    if NET_PARAMS['do_fit']:
        log('Training the model...')
        model.fit(data['x_train'], data['y_train'],
                  epochs=NET_PARAMS['epochs'],
                  batch_size=NET_PARAMS['batch_size'],
                  shuffle=True,
                  verbose=0,
                  validation_split=0.1,
                  callbacks=[ModelCheckpoint(best_model_path, monitor='val_loss', verbose=False, save_best_only=True,
                                             save_weights_only=False)])
        log('Model trained.')

    # -- NEURAL NETWORK EVALUATION

    # Evaluate on test data (test data = train data - insufficient amount of data)
    evaluate_model(best_model_path, data['x_train'], data['y_train'], target_names, print_detail=False)
