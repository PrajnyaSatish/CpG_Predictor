import torch
from torch.nn import Module, Linear, LSTM, Embedding, Dropout

class CpGPredictor(Module):
    """
    Simple model that uses a LSTM to count the number of CpGs in a sequence
    """

    def __init__(self, num_layers, hidden_layer_dim, dropout_p, vocab_size, embed_dim):
        super(CpGPredictor, self).__init__()
        self.vocab_size = vocab_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers
        self.hidden_layer_dim = hidden_layer_dim
        self.embed_dim = embed_dim

        ## Define the model layers
        self.embedding = Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_size=hidden_layer_dim,
                                  num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_layer_dim, 1)
        self.lstm2 = LSTM(hidden_layer_dim, hidden_size=hidden_layer_dim, batch_first=True)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        yhat, _ = self.lstm(embedded)

        ## Without Dropout, the model overfits too soon
        yhat = self.dropout1(yhat)
        yhat, _ = self.lstm2(yhat)
        yhat = self.dropout2(yhat)
        out = self.linear(yhat[:, -1, :])
        return out