from intent_classifier_pytorch import *


class Data(Dataset):
    def __init__(self):
        self.x_train = torch.from_numpy(data['x_train'])
        self.y_train = torch.from_numpy(data['y_train'])
        self.len = self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':

    # -- DATA PREPROCESSING
    DATA_FILE = 'data.txt'  # source txt file

    # Read data and form samples vectors using BERT
    samples, targets, target_names = read_data(data_file=DATA_FILE)

    y1 = []
    y2 = []

    for i in range(10):
        # Form data for Pytorch
        torch.manual_seed(i)
        data, target_numbers = form_data(samples, targets, target_names)
        data_set = Data()

        # -- SETTINGS
        NET_PARAMS = {
            'hidden_layers': [58, 28],
            'learning_rate': 0.1,
            'epochs': 50,
            'batch_size': 10,
            'do_fit': True,
            'overwrite_best_model': False
        }
        neurons_num = {'inp_shape': len(data['x_train'][0]), 'hidden_layers': NET_PARAMS['hidden_layers'],
                       'out_units': len(target_names)}
        with open('temp/neurons_num.pickle', 'wb') as wb:  # Store data (serialize) for model loading
            pickle.dump(neurons_num, wb, protocol=pickle.HIGHEST_PROTOCOL)

        # -- NEURAL NETWORK DESIGN

        # Create Data Loader
        trainloader = DataLoader(dataset=data_set, batch_size=NET_PARAMS['batch_size'])

        # Network architecture
        model = design_model(inp_shape=neurons_num['inp_shape'], hidden_layers=neurons_num['hidden_layers'],
                             out_units=neurons_num['out_units'], print_summary=False)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=NET_PARAMS['learning_rate'])

        # -- NEURAL NETWORK TRAINING

        # Train the network
        if NET_PARAMS['do_fit']:
            log('Training the model...')
            accu, losses = fit_model(model, trainloader, NET_PARAMS, optimizer, criterion, verbose=0)
            log('Model trained.')
            y1.append(accu)
            y2.append(losses)

    with open('temp/pytorch/y1.pickle', 'wb') as wb:
        pickle.dump(y1, wb, protocol=pickle.HIGHEST_PROTOCOL)
    with open('temp/pytorch/y2.pickle', 'wb') as wb:
        pickle.dump(y2, wb, protocol=pickle.HIGHEST_PROTOCOL)
