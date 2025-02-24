REGISTRY = {}

from .rnn_agent import RNNAgent
from .cnn_agent import CNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["cnn"] = CNNAgent
