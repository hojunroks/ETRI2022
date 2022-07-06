import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer

## TODO: develop Linear model (FC*1, FC*2)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, bidirectional_flag, dropout=0.2, emo_classes=1, device=0):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_emotion_classes = emo_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional_flag = bidirectional_flag
        self.dropout = dropout
        self.device = device
        #self.seq_length = seq_length
        
        if self.bidirectional_flag == True:
            self.D = 2
        else:
            self.D = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional_flag, dropout=self.dropout)
        
        self.fc_act = nn.Linear(hidden_size*self.D, num_classes)
        self.fc_emotion = nn.Linear(hidden_size*self.D, self.num_emotion_classes)
        


    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.D * self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        c_0 = Variable(torch.zeros(
            self.D * self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
#         h_out = h_out.view(-1, self.hidden_size)
        if self.bidirectional_flag == True:
            ula = ula.view(-1, self.hidden_size*2)
        else:
            ula = ula.view(-1, self.hidden_size)

        out_act = self.fc_act(ula)
        out_emotion = self.fc_emotion(ula)
        out = out_act, out_emotion
        return out


class MLP(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout=0.5, emo_classes=1):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        for i in range(num_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(
            *layers
        )
        self.action = nn.Linear(hidden_size, num_classes)
        self.emotion = nn.Linear(hidden_size, emo_classes)

    def forward(self, x):
        x = self.encoder(x)
        out_act = self.action(x)
        out_emotion = self.emotion(x)
        return out_act, out_emotion

class TransformerBased(nn.Module):
    def __init__(self, input_size: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int, num_actions: int, dropout: float = 0.5, num_emo_classes: int = 1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(input_size, d_model)
        self.d_model = d_model
        self.decoder_action = nn.Linear(d_model, num_actions)
        self.decoder_emotion = nn.Linear(d_model, num_emo_classes)

    def forward(self, src, src_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, input_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output_action = self.decoder_action(output).squeeze(dim=1)
        # print(output_action.shape)
        output_emotion = self.decoder_emotion(output).squeeze(dim=1)
        # print(output_emotion.shape)
        return output_action, output_emotion


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)