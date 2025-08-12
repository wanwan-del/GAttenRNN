from models.AttenRNN.attention_rnn import AttenRNN
from models.DynamicNet.dynamic_net_multi import DynamicNet_multi
from models.DynamicNet.dynamic_net_single import DynamicNet_sigle
from models.RNN.rnn import RNN

MODELS = {
    "DynamicNet_single": DynamicNet_sigle,
    "DynamicNet_multi": DynamicNet_multi,
    "AttenRNN": AttenRNN,
    "RNN": RNN,
}