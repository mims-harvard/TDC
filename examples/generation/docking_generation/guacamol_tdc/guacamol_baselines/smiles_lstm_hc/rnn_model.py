import torch
import torch.nn as nn


class SmilesRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rnn_dropout) -> None:
        """
            Basic RNN language model for SMILES

        Args:
            input_size: number of input symbols
            hidden_size: number of hidden units
            output_size: number of output symbols
            n_layers: number of hidden layers
            rnn_dropout: recurrent dropout
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=rnn_dropout)
        self.init_weights()

    def init_weights(self):
        # encoder / decoder
        nn.init.xavier_uniform_(self.encoder.weight)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)

        # RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # LSTM remember gate bias should be initialised to 1
                # https://github.com/pytorch/pytorch/issues/750
                r_gate = param[int(0.25 * len(param)):int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, x, hidden):
        embeds = self.encoder(x)
        output, hidden = self.rnn(embeds, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, bsz, device):
        # LSTM has two hidden states...
        return (torch.zeros(self.n_layers, bsz, self.hidden_size).to(device),
                torch.zeros(self.n_layers, bsz, self.hidden_size).to(device))

    @property
    def config(self):
        return dict(input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    n_layers=self.n_layers,
                    rnn_dropout=self.rnn_dropout)
