import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# import scikitplot as skplt
import torch
import numpy as np
import os
import torch.nn.functional as F
from data_prep import *
from logs import log, fix_ts

torch.manual_seed(0)

# Force use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -- SETTINGS

DATA_FILE = 'data.txt'  # source txt file


NET_PARAMS = {
    'hidden_layers': [15],
    'learning_rate': 0.1,
    'epochs': 21,
    'batch_size': 10,
    'do_fit': True,
    'overwrite_best_model': True
}

data = {}

def form_data(samples, targets, target_names):
    global data
    """ Prepare matrices for machine learning in Keras
        return dict with keys: 'x_train', 'y_train'
    """

    data = {}

    x = np.array(samples)
    data['x_train'] = x.reshape(-1, x.shape[1]).astype('float32')

    target_numbers = []
    for target in targets:
        target_numbers.append(target_names.index(target))
    data['y_train'] = F.one_hot(torch.tensor(target_numbers), num_classes=len(target_names)).numpy()
    log('Data formed')

    return data, target_numbers


def design_model(inp_shape, hidden_layers, out_units, print_summary=True):
    global data
    log('Building the model...')

    model = Net(inp_shape, hidden_layers, out_units)

    if print_summary:
        summary(model, (1, len(data['x_train'][0])))

    return model


def fit_model(model, trainloader, params, optimizer, criterion, verbose=1):
    train_accu = []
    train_losses = []

    for epoch in range(params['epochs']):
        # Set model to train configuration
        model.train()

        running_loss = 0
        correct = 0
        total = 0
        for x_train, y_train in trainloader:
            # Convert to FloatTensor
            y_train = y_train.type(torch.FloatTensor)

            # Clear gradient
            optimizer.zero_grad()

            # Make a prediction
            y_pred = model(x_train)

            # Calculate loss
            loss = criterion(y_pred, y_train)

            # Calculate gradients of parameters
            loss.backward()

            # Update parameters
            optimizer.step()

            running_loss += loss.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train.max(1)[1]).sum().item()

        # Loss for each epoch = running_loss/ number of batches
        train_loss = running_loss / len(trainloader)

        # Accuracy = number of correct classifications / the total amount of classifications
        accu = correct / total

        train_accu.append(accu)
        train_losses.append(train_loss)

        if verbose > 0:
            print('Epoch: %d/%d loss: %.4f - accuracy: %.4f' % (epoch + 1, params['epochs'], train_loss, accu))
    return train_accu, train_losses


def evaluate_model(model_path, x_test, y_test, print_detail):
    # Load model for evaluation
    model = load_best_model(model_path)

    # Evaluate on test data
    y_pred = model(torch.from_numpy(x_test))

    y_pred_num = torch.max(y_pred.data, 1).indices
    y_test_num = torch.max(torch.from_numpy(y_test), 1).indices
    print("Model accuracy: {}".format(accuracy_score(y_test_num, y_pred_num)))
    # skplt.metrics.plot_confusion_matrix(
    #     y_test_num, y_pred_num, x_tick_rotation=90)
    # plt.show()

    if print_detail:  # Print predicted vs test classes
        # Load target names
        with open('temp/target_names.pickle', 'rb') as rb:
            target_names = pickle.load(rb)
        y_pred_names = [target_names[i] for i in y_pred_num]
        y_test_names = [target_names[i] for i in y_test_num]

        for i in range(len(y_test_num)):
            print(
                'Predicted: {} {}   Test: {} {}'.format(y_pred_num[i], y_pred_names[i], y_test_num[i], y_test_names[i]))
    # save wrong predictions to tsv file
    with open('temp/target_names.pickle', 'rb') as rb:
        target_names = pickle.load(rb)
    with open(DATA_FILE, 'r', encoding='utf-8-sig') as r:
        sentences = r.readlines()
    with open('wrong_predictions.tsv', 'w', encoding='utf-8-sig') as w:
        w.write('Sentence\tPredicted class\tCorrect class\n')
        for i in range(len(y_test_num)):
            pred = y_pred_num[i]
            true = y_test_num[i]
            if pred != true:
                sentence = sentences[i][:-1].split(' ')[1]
                print('{}\t Predicted class: {} Correct class: {}'.format(sentence, target_names[pred], target_names[true]))
                w.write('{}\t{}\t{}\n'.format(sentence, target_names[pred], target_names[true]))


def load_best_model(model_path="models/best"):
    with open('temp/neurons_num.pickle', 'rb') as rb:  # Load data (deserialize)
        neurons_num = pickle.load(rb)
    model = Net(inp_shape=neurons_num['inp_shape'], hidden_layers=neurons_num['hidden_layers'],
                out_units=neurons_num['out_units'])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_single(user_input, model):
    """
    Predict class for given input

    """
    target_names = get_target_names()

    # Form the text for classification
    user_input = read_sentence(user_input)
    user_input = np.array([user_input])
    user_input = user_input.reshape(-1, user_input.shape[1]).astype('float32')
    user_input = torch.from_numpy(user_input)

    # Make prediction
    pred = model(user_input)
    pred = pred.argmax(axis=1)

    return target_names[pred]


# def train_process_graph(accu, losses):
#     # Creating plot with loss
#     fig, ax1 = plt.subplots()
#
#     color = 'tab:red'
#     ax1.set_xlabel('Epochs')
#     ax1.set_ylabel('Loss', color=color)
#     ax1.plot(losses, color=color)
#     ax1.tick_params(axis='y', labelcolor=color)
#
#     # Adding Twin Axes to plot using accuracy
#     ax2 = ax1.twinx()
#
#     color = 'tab:green'
#     ax2.set_ylabel('Accuracy', color=color)
#     ax2.plot(accu, color=color)
#     ax2.tick_params(axis='y', labelcolor=color)
#
#     # Adding title
#     plt.title('Training process')
#
#     # Show plot
#     plt.show()


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

    torch.manual_seed(0)

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
        if retrain_all: # sample will be added to the training set
            with open(DATA_FILE, 'a+', encoding='utf-8-sig') as fp:
                new_data = "\n{} {}".format(correct_intent.upper(), inputs_and_predictions["prev"]["input"])
                print('New data: {}'.format(new_data))
                fp.write(new_data)
            # call main function
            main()
        else: # retrain model on the new sample only
            target = target_names.index(correct_intent)
            target = torch.tensor([target])
            target = F.one_hot(target, num_classes=len(target_names))
            user_input = np.array([user_input])
            user_input = user_input.reshape(-1, user_input.shape[1]).astype('float32')
            user_input = torch.from_numpy(user_input)
            model = load_best_model()
            # Retrain model
            optimizer = torch.optim.SGD(model.parameters(), lr=NET_PARAMS['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            model.train()
            optimizer.zero_grad()
            y_pred = model(user_input)
            y_train = target.type(torch.FloatTensor)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            print('Model retrained')
            torch.save(model.state_dict(), 'models/best')
    else: # target does not exist
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
                incorrect_intent = target_names.index(inputs_and_predictions["prev"]["intent"])
                # create pytorch tensor with 0.5 probabilities for all intents except for the incorrect one
                target = np.array([0.5] * len(target_names))
                target[incorrect_intent] = 0

                # Read data and form samples vectors using BERT
                samples, targets, target_names = read_data(data_file=DATA_FILE)

                # Form data for Pytorch
                data, target_numbers = form_data(samples, targets, target_names)
                data['y_train'] = np.concatenate((data['y_train'], [target]))
                data['x_train'] = np.concatenate((data['x_train'], [user_input]))
                data_set = Data()

                neurons_num = {'inp_shape': len(data['x_train'][0]), 'hidden_layers': NET_PARAMS['hidden_layers'],
                               'out_units': len(target_names)}
                with open('temp/neurons_num.pickle', 'wb') as wb:  # Store data (serialize) for model loading
                    pickle.dump(neurons_num, wb, protocol=pickle.HIGHEST_PROTOCOL)

                trainloader = DataLoader(dataset=data_set, batch_size=NET_PARAMS['batch_size'])
                model = design_model(inp_shape=neurons_num['inp_shape'], hidden_layers=neurons_num['hidden_layers'],
                                     out_units=neurons_num['out_units'])
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=NET_PARAMS['learning_rate'])
                best_model_path = "models/best"
                log('Training the model...')
                accu, losses = fit_model(model, trainloader, NET_PARAMS, optimizer, criterion, verbose=0)
                torch.save(model.state_dict(), best_model_path)
                log('Model trained.')
                # train_process_graph(accu, losses)

                evaluate_model(best_model_path, data['x_train'], data['y_train'], print_detail=False)
            else: # retrain model on the new sample only
                inccorect_intent = target_names.index(inputs_and_predictions["prev"]["intent"])
                # create pytorch tensor with 0.5 probabilities for all intents except for the incorrect one
                target = np.array([0.5] * len(target_names))
                target[inccorect_intent] = 0
                target = np.array([target])
                target = torch.from_numpy(target)
                user_input = np.array([user_input])
                user_input = user_input.reshape(-1, user_input.shape[1]).astype('float32')
                user_input = torch.from_numpy(user_input)
                model = load_best_model()
                # Retrain model
                optimizer = torch.optim.SGD(model.parameters(), lr=NET_PARAMS['learning_rate'])
                criterion = nn.CrossEntropyLoss()
                model.train()
                optimizer.zero_grad()
                y_pred = model(user_input)
                y_train = target.type(torch.FloatTensor)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                print('Model retrained')
                torch.save(model.state_dict(), 'models/best')


class Data(Dataset):
    global data
    def __init__(self):
        self.x_train = torch.from_numpy(data['x_train'])
        self.y_train = torch.from_numpy(data['y_train'])
        self.len = self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len


class Net(nn.Module):
    def __init__(self, inp_shape, hidden_layers, out_units):
        super(Net, self).__init__()
        self.NET_I = nn.Linear(inp_shape, hidden_layers[0])
        self.NET_O = nn.Linear(hidden_layers[0], out_units)

    def forward(self, x):
        x = torch.sigmoid(self.NET_I(x))
        x = self.NET_O(x)
        return x


def main():
    global data
    # -- DATA PREPROCESSING
    # Read data and form samples vectors using BERT
    samples, targets, target_names = read_data(data_file=DATA_FILE)

    # Form data for Pytorch
    data, target_numbers = form_data(samples, targets, target_names)
    data_set = Data()


    neurons_num = {'inp_shape': len(data['x_train'][0]), 'hidden_layers': NET_PARAMS['hidden_layers'],
                   'out_units': len(target_names)}
    with open('temp/neurons_num.pickle', 'wb') as wb:  # Store data (serialize) for model loading
        pickle.dump(neurons_num, wb, protocol=pickle.HIGHEST_PROTOCOL)

    # -- NEURAL NETWORK DESIGN

    # Create Data Loader
    trainloader = DataLoader(dataset=data_set, batch_size=NET_PARAMS['batch_size'])

    # Network architecture
    model = design_model(inp_shape=neurons_num['inp_shape'], hidden_layers=neurons_num['hidden_layers'],
                         out_units=neurons_num['out_units'])

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=NET_PARAMS['learning_rate'])

    # -- NEURAL NETWORK TRAINING

    # Save the best gained model to this path
    if NET_PARAMS['overwrite_best_model']:
        best_model_path = "models/best"
    else:
        best_model_path = "models/model_{}".format(fix_ts().replace(':', '-'))

    # Train the network
    if NET_PARAMS['do_fit']:
        log('Training the model...')
        accu, losses = fit_model(model, trainloader, NET_PARAMS, optimizer, criterion, verbose=0)
        torch.save(model.state_dict(), best_model_path)
        log('Model trained.')
        # train_process_graph(accu, losses)

    # -- NEURAL NETWORK EVALUATION

    # Evaluate on test data (test data = train data - insufficient amount of data)
    evaluate_model(best_model_path, data['x_train'], data['y_train'], print_detail=True)

if __name__ == '__main__':
    main()

