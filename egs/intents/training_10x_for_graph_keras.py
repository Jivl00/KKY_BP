from intent_classifier_keras import *

if __name__ == '__main__':

    log('TF loaded. Using ' + ('GPU' if 'GPU' in str(device_lib.list_local_devices()) else 'CPU'))

    # -- DATA PREPROCESSING
    DATA_FILE = 'data.txt'  # source txt file

    # Read data and form samples vectors using BERT
    samples, targets, target_names = read_data(data_file=DATA_FILE)

    y1 = []
    y2 = []

    for i in range(10):
        tf.random.set_seed(i)
        random.seed(i)

        # Form data for Keras
        data, target_numbers = form_data(samples, targets, target_names)

        # -- SETTINGS

        NET_PARAMS = {
            'hidden_layers': [58, 28],
            'learning_rate': 0.1,
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy'],
            'epochs': 50,
            'batch_size': 10,
            'do_fit': True,
            'overwrite_best_model': False
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

        # Train the network
        if NET_PARAMS['do_fit']:
            log('Training the model...')
            h = model.fit(data['x_train'], data['y_train'],
                          epochs=NET_PARAMS['epochs'],
                          batch_size=NET_PARAMS['batch_size'],
                          shuffle=True,
                          verbose=0,
                          validation_split=0.1)
            log('Model trained.')
            y1.append(h.history['accuracy'])
            y2.append(h.history['loss'])

    with open('temp/keras/y1.pickle', 'wb') as wb:
        pickle.dump(y1, wb, protocol=pickle.HIGHEST_PROTOCOL)
    with open('temp/keras/y2.pickle', 'wb') as wb:
        pickle.dump(y2, wb, protocol=pickle.HIGHEST_PROTOCOL)
