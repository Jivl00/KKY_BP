import os
import pickle

from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"]=""
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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



    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        for i in range(len(self.hidden_layers) - 1):
            x = torch.sigmoid(getattr(self, 'fc' + str(i + 2))(x))
        x = self.fc_out(x)
        return x



def load_best_model(model_path="models/best_model"):
    with open('models/neurons_num.pickle', 'rb') as rb:  # Load data (deserialize)
        neurons_num = pickle.load(rb)
    model = Net(inp_shape=neurons_num['inp_shape'], hidden_layers=neurons_num['hidden_layers'],
                out_units=neurons_num['out_shape'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model.to(device)


class Model:

    def __init__(self):
        self.text2vec = SentenceTransformer('fav-kky/FERNET-C5')

        # Net
        self.net = None

        # Labels
        self.target2label = None

        # Optimizer
        self.optimizer = None

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def encode_input(self, text):
        return self.text2vec.encode(text)

    def decode_output(self, vec):
        return self.target2label[torch.argmax(vec).item()]

    def init_net(self):
        self.net = load_best_model()
        self.net.eval()


bert_path = 'fav-kky/FERNET-C5'
bert_model = SentenceTransformer(bert_path)

def get_target_names():
    # Load data (deserialize)
    with open('models/target_names.pickle', 'rb') as rb:
        unserialized_data = pickle.load(rb)
    return list(unserialized_data)

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
    user_input = user_input.reshape(-1, user_input.shape[1]).astype('float32')
    user_input = torch.from_numpy(user_input)

    # Make prediction
    pred = model(user_input)
    pred = pred.argmax(axis=1)

    return target_names[pred]

# model = load_best_model()
# print(predict_single('jak řeknu sbohem v čínštině', model))


