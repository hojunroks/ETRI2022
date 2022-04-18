import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, bidirectional_flag, emo_classes=1):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_emotion_classes = emo_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional_flag = bidirectional_flag
        #self.seq_length = seq_length
        
        if self.bidirectional_flag == True:
            self.D = 2
        else:
            self.D = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional_flag, dropout=0.2)
        
        self.fc_act = nn.Linear(hidden_size*self.D, num_classes)
        self.fc_emotion = nn.Linear(hidden_size*self.D, self.num_emotion_classes)


    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.D * self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.D * self.num_layers, x.size(0), self.hidden_size))
        
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