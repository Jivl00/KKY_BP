import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
from transformers import logging

torch.manual_seed(0)
logging.set_verbosity_error()  # shut up Bert warning

NET_PARAMS = {
    'hidden_layers': [512, 256],
    'learning_rate': 0.003,
    'learning_rate_decay': 0.9,
    'batch_size': 2048,
    'epochs': 300
}


class Net(nn.Module):
    def __init__(self, inp_shape, hidden_layers, out_units):
        super(Net, self).__init__()
        self.hidden_layers = hidden_layers
        self.fc1 = nn.Linear(inp_shape, hidden_layers[0])
        for i in range(len(hidden_layers) - 1):
            setattr(self, 'fc' + str(i + 2), nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        self.fc_out = nn.Linear(hidden_layers[-1], out_units)

        self.train()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        for i in range(len(self.hidden_layers) - 1):
            x = torch.sigmoid(getattr(self, 'fc' + str(i + 2))(x))
        x = self.fc_out(x)
        return x

    def new_m(self, m):  # m is the new number of classes
        self.fc_out = nn.Linear(self.hidden_layers[-1], m)


class Model:
    def __init__(self):
        # Net
        self.net = None

        # Optimizer
        self.optimizer = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def init_net(self, inp_shape, hidden_layers, out_units, print_summary=False):
        self.net = Net(inp_shape, hidden_layers, out_units)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=NET_PARAMS['learning_rate'])

        if print_summary:
            summary(self.net, (inp_shape,))

    def reinit_net(self, m, print_summary=False):
        self.net.new_m(m)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=NET_PARAMS['learning_rate'])

        if print_summary:
            summary(self.net, (self.net.fc1.in_features,))

    def fit(self, trainloader, validloader, epochs, verbose=0):
        train_accu = []
        train_losses = []
        dev_losses = []
        prev_valid_loss = np.inf
        val_loss_increase = 0

        for epoch in range(epochs):
            self.net.train()

            if epoch % 50 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr'] * NET_PARAMS['learning_rate_decay']

            running_loss = 0.0
            correct = 0
            total = 0

            for x_train, y_train in trainloader:
                y_train = y_train.type(torch.FloatTensor)
                self.optimizer.zero_grad()
                y_pred = self.net(x_train)
                loss = self.criterion(y_pred, y_train)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(y_pred.data, 1)
                total += y_train.size(0)
                correct += predicted.eq(y_train.max(1)[1]).sum().item()
            epoch_loss = running_loss / len(trainloader)
            accu = correct / total
            train_accu.append(accu)
            train_losses.append(epoch_loss)
            if verbose and epoch % 50 == 0:
                print('Epoch: {} - loss: {:.4f} - acc: {:.4f}'.format(epoch + 1, epoch_loss, accu))

            valid_loss = 0.0
            self.net.eval()
            for x_valid, y_valid in validloader:
                y_valid = y_valid.type(torch.FloatTensor)
                y_pred = self.net(x_valid)
                loss = self.criterion(y_pred, y_valid)
                valid_loss += loss.item()
            valid_loss /= len(validloader)
            dev_losses.append(valid_loss)
            if verbose and epoch % 50 == 0:
                print('Validation loss: {:.4f}'.format(valid_loss))

            if valid_loss > prev_valid_loss:
                val_loss_increase += 1
            else:
                val_loss_increase = 0
            if val_loss_increase > NET_PARAMS['epochs'] / 5:
                print('Early stopping at epoch {}'.format(epoch))
                break
            prev_valid_loss = valid_loss

        return train_accu, train_losses, dev_losses

    def evaluate(self, testloader):
        self.net.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        y_test_nums = []
        y_pred_nums = []
        for x_test, y_test in testloader:
            y_test = y_test.type(torch.FloatTensor)
            y_pred = self.net(x_test)
            loss = self.criterion(y_pred, y_test)
            test_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += y_test.size(0)
            correct += predicted.eq(y_test.max(1)[1]).sum().item()
            y_test_nums += y_test.max(1)[1].tolist()
            y_pred_nums += predicted.tolist()
        test_loss /= len(testloader)
        accu = correct / total
        print('Test loss: {:.4f} - acc: {:.4f}'.format(test_loss, accu))
        return accu, test_loss


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
    samples = x.reshape(-1, x.shape[1]).astype('float32')

    target_numbers = []
    for target in targets:
        target_numbers.append(list(target_names).index(target))
    targets = F.one_hot(torch.tensor(target_numbers), num_classes=len(target_names)).numpy()
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
    plt.savefig('train_process_pytorch_big.pdf')
    plt.show()


class IntentDataset(Dataset):
    def __init__(self, samples, targets):
        self.samples = torch.from_numpy(samples)
        self.targets = torch.from_numpy(targets)
        self.len = self.samples.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]
        return sample, target


def main():
    numers_of_samples_per_class = [{'train': 100, 'test': 30, 'valid': 20},
                                   {'train': 80, 'test': 24, 'valid': 16},
                                   {'train': 60, 'test': 18, 'valid': 12},
                                   {'train': 40, 'test': 12, 'valid': 8},
                                   {'train': 20, 'test': 6, 'valid': 4}]

    classes_cut = [{'train': 100 * 0, 'test': 30 * 0, 'valid': 20 * 0},
                   {'train': 100 * 30, 'test': 30 * 30, 'valid': 20 * 30},
                   {'train': 100 * 60, 'test': 30 * 60, 'valid': 20 * 60},
                   {'train': 100 * 90, 'test': 30 * 90, 'valid': 20 * 90},
                   {'train': 100 * 120, 'test': 30 * 120, 'valid': 20 * 120}]
    for numer_of_samples_per_class in numers_of_samples_per_class:
        for classes in classes_cut:

            NET_PARAMS = {
                'hidden_layers': [512, 256],
                'learning_rate': 0.001,
                'epochs': 150,
                'batch_size': 5000,
                'do_fit': True,
                'overwrite_best_model': True
            }
            # Load data
            path = 'dataset-2'
            with open(path + '/train-cs.tsv', 'r', encoding='utf-8-sig') as f:
                train = [line for line in f]
                train = train[classes['train']:]
                new_train = []
                for i in range(0, len(train), 100):
                    # keep only number_of_samples_per_class samples per class
                    new_train.extend(train[i:i + numer_of_samples_per_class['train']])
                train = new_train

            with open(path + '/test-cs.tsv', 'r', encoding='utf-8-sig') as f:
                test = [line for line in f]
                test = test[classes['test']:]
                # new_test = []
                # for i in range(0, len(test), 30):
                #     # keep only number_of_samples_per_class samples per class
                #     new_test.extend(test[i:i + numer_of_samples_per_class['test']])
                # test = new_test
            with open(path + '/dev-cs.tsv', 'r', encoding='utf-8-sig') as f:
                valid = [line for line in f]
                valid = valid[classes['valid']:]
                # new_valid = []
                # for i in range(0, len(valid), 20):
                #     # keep only number_of_samples_per_class samples per class
                #     new_valid.extend(valid[i:i + numer_of_samples_per_class['valid']])
                # valid = new_valid
            print('Testing with {} traning samples, {} test samples and {} validation samples.'.format(len(train),
                                                                                                       len(test),
                                                                                                       len(valid)))
            samples, targets, target_names = read_data(train, test, valid)

            train_dataset = IntentDataset(samples['train'], targets['train'])
            test_dataset = IntentDataset(samples['test'], targets['test'])
            valid_dataset = IntentDataset(samples['valid'], targets['valid'])

            neurons_num = {'inp_shape': len(samples['valid'][0]), 'out_shape': len(target_names)}

            train_loader = DataLoader(dataset=train_dataset, batch_size=NET_PARAMS['batch_size'])
            test_loader = DataLoader(dataset=test_dataset, batch_size=NET_PARAMS['batch_size'])
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=NET_PARAMS['batch_size'])

            model = Model()
            model.init_net(neurons_num['inp_shape'], NET_PARAMS['hidden_layers'], neurons_num['out_shape'])

            print('Training the model...')
            train_accu, train_losses, dev_losses = model.fit(train_loader, valid_loader, epochs=NET_PARAMS['epochs'],
                                                             verbose=0)
            # train_process_graph(train_accu, train_losses)
            print('Model trained.')

            print('Testing the model...')

            eval_accu, eval_loss = model.evaluate(test_loader)
            with open('analysis2_new.txt', 'a', encoding='utf-8') as f:
                f.write("{}\t{}\t{}\t{}\n".format(len(target_names), numer_of_samples_per_class['train'], eval_accu,
                                                  eval_loss))


if __name__ == '__main__':
    main()
