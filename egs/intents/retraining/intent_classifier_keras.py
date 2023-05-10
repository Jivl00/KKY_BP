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
from data_prep import *
from logs import log, fix_ts

tf.random.set_seed(0)
random.seed(0)


# Force use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

# NET_PARAMS = {
#     'hidden_layers': [58, 28],
#     'learning_rate': 0.1,
#     'loss': 'categorical_crossentropy',
#     'metrics': ['accuracy'],
#     'epochs': 50,
#     'batch_size': 10,
#     'do_fit': True,
#     'overwrite_best_model': True
# }

DATA_FILE = 'data.txt'  # source txt file


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


def load_best_model(best_model_path="models/best.h5"):
    return load_model(best_model_path)


def train_process_graph(accu, losses):
    # Creating plot with loss
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
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

    # Show plot
    plt.show()

def retrain(inputs_and_predictions, correct_intent, add_intent=False, retrain_all=True):
    """
    Retrain the model with new data

    params:
    inputs_and_predictions: dict with user inputs and predictions
    correct_intent: correct intent
    add_intent: if True, new intent will be added to the training set
    retrain_all: if True, the model will be retrained from scratch - must be True if add_intent is True

    """
    print('Retraining model...')
    tf.random.set_seed(0)
    random.seed(0)

    # Load and form user input
    user_input = inputs_and_predictions["prev"]["input"]
    print('User input: {}'.format(user_input))
    user_input = read_sentence(user_input)

    # Load and form target
    correct_intent = correct_intent.lower()

    # Load target names
    target_names = get_target_names()
    if correct_intent in target_names: # target already exists
        print('Target already exists')
        if retrain_all: # sample will be added to the training data
            with open(DATA_FILE, 'a+', encoding='utf-8-sig') as fp:
                new_data = "\n{} {}".format(correct_intent.upper(), inputs_and_predictions["prev"]["input"])
                print('New data: {}'.format(new_data))
                fp.write(new_data)
            # call main function
            main()
        else: # retrain model on the new sample only
            target = target_names.index(correct_intent)
            target = keras.utils.to_categorical(target, num_classes=len(target_names))
            target = np.array([target])
            user_input = np.array([user_input])
            model = load_best_model()
            model.fit(user_input, target, epochs=1, batch_size=1, verbose=1)
            model.save('models/best.h5')
    else: # new target
        if add_intent:
            if not retrain_all:
                print("retrain_all must be True if add_intent is True")
                return
            with open(DATA_FILE, 'a+', encoding='utf-8-sig') as fp:
                new_data = "\n{} {}".format(correct_intent.upper(), inputs_and_predictions["prev"]["input"])
                print('New data: {}'.format(new_data))
                fp.write(new_data)
            # call main function
            main()
        else:
            if retrain_all:
                incorrect_target = target_names.index(inputs_and_predictions["prev"]["intent"])
                # create keras tensor with 0.5 values
                target = np.full((1, len(target_names)), 0.5, dtype=np.float32)
                # set incorrect target to 0
                target[0][incorrect_target] = 0

                # Read data and form samples vectors using BERT
                samples, targets, target_names = read_data(data_file=DATA_FILE)
                data, target_numbers = form_data(samples, targets, target_names)
                data['y_train'] = np.concatenate((data['y_train'], target))
                data['x_train'] = np.concatenate((data['x_train'], [user_input]))

                model = design_model(inp_shape=len(data['x_train'][0]), out_units=len(target_names), params=NET_PARAMS)
                optimizer = tf.keras.optimizers.SGD(learning_rate=NET_PARAMS['learning_rate'])
                model.compile(loss=NET_PARAMS['loss'], optimizer=optimizer, metrics=NET_PARAMS['metrics'])
                best_model_path = "models/best.h5"
                log('Training the model...')
                h = model.fit(data['x_train'], data['y_train'],
                              epochs=NET_PARAMS['epochs'],
                              batch_size=NET_PARAMS['batch_size'],
                              shuffle=True,
                              verbose=0,
                              validation_split=0.1,
                              callbacks=[
                                  ModelCheckpoint(best_model_path, monitor='val_loss', verbose=False,
                                                  save_best_only=True,
                                                  save_weights_only=False)])
                log('Model trained.')
                train_process_graph(h.history['accuracy'], h.history['loss'])

                evaluate_model(best_model_path, data['x_train'], data['y_train'], target_names, print_detail=False)

            else: # retrain model on the new sample only
                incorrect_target = target_names.index(inputs_and_predictions["prev"]["intent"])
                # create keras tensor with 0.5 values
                target = np.full((1, len(target_names)), 0.5, dtype=np.float32)
                # set incorrect target to 0
                target[0][incorrect_target] = 0
                target = np.array(target)
                user_input = np.array([user_input])
                model = load_best_model()
                model.fit(user_input, target, epochs=1, batch_size=1, verbose=1)
                model.save('models/best.h5')


def main():

    log('TF loaded. Using ' + ('GPU' if 'GPU' in str(device_lib.list_local_devices()) else 'CPU'))

    # -- DATA PREPROCESSING

    # Read data and form samples vectors using BERT
    samples, targets, target_names = read_data(data_file=DATA_FILE)
    # print('Targets: {}'.format(targets))
    # print('Target names: {}'.format(target_names))

    # Form data for Keras
    data, target_numbers = form_data(samples, targets, target_names)
    # print('Data x: {}'.format(data['x_train']))
    # print('Data y: {}'.format(data['y_train']))
    #
    # print('Target numbers: {}'.format(len(target_numbers)))

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
        best_model_path = "models/best.h5"
    else:
        best_model_path = "models/model_{}.h5".format(fix_ts().replace(':', '-'))

    # Train the network
    if NET_PARAMS['do_fit']:
        log('Training the model...')
        h = model.fit(data['x_train'], data['y_train'],
                      epochs=NET_PARAMS['epochs'],
                      batch_size=NET_PARAMS['batch_size'],
                      shuffle=True,
                      verbose=0,
                      validation_split=0.1,
                      callbacks=[
                          ModelCheckpoint(best_model_path, monitor='val_loss', verbose=False, save_best_only=True,
                                          save_weights_only=False)])
        log('Model trained.')
        train_process_graph(h.history['accuracy'], h.history['loss'])

    # -- NEURAL NETWORK EVALUATION

    # Evaluate on test data (test data = train data - insufficient amount of data)
    evaluate_model(best_model_path, data['x_train'], data['y_train'], target_names, print_detail=False)

if __name__ == '__main__':
    main()