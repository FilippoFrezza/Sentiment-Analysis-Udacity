import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers, dropout=0.2):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()
        
        #self.n_layers = n_layers
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        
        out = self.dropout(lstm_out)
        out = self.dense(out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
    
