import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAgent(nn.Module):
    def __init__(self, input_channel, args):
        super(CNNAgent, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(input_channel, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 64)
        self.fc3 = nn.Linear(64, args.n_actions)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, x, hidden_state):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = F.relu(self.fc2(h))
        q = self.fc3(q)
        return q, h


def test():
    from types import SimpleNamespace
    arg = SimpleNamespace()
    arg.rnn_hidden_dim = 128
    arg.n_actions = 9
    net = CNNAgent(input_channel=3, args=arg)
    state = torch.rand(1, 3, 16, 16)
    h = torch.rand(1, 128)
    out = net.forward(state, h)
    print(out[0].shape)
    print(out[1].shape)


if __name__ == "__main__":
    test()

