import torch.nn as nn


class RNN_instruction_encoder(nn.Module):
    def __init__(self, vocab_size, word_vec_dim, hidden_size,
                 n_layers, input_dropout_p=0, dropout_p=0, bidirectional=True,
                 variable_lengths=True, word2vec=None, fix_embeddings=False,
                 rnn_cell='lstm'):
        super(RNN_instruction_encoder, self).__init__()
        assert rnn_cell in ['lstm', 'gru']
        self.variable_lengths = variable_lengths
        if word2vec is not None:
            assert word2vec.size(0) == vocab_size
            self.word_vec_dim = word2vec.size(1)
            self.embedding = nn.Embedding(vocab_size, self.word_vec_dim)
            self.embedding.weight = nn.Parameter(word2vec)
        else:
            self.word_vec_dim = word_vec_dim
            self.embedding = nn.Embedding(vocab_size, word_vec_dim)
        if fix_embeddings:
            self.embedding.weight.requires_grad = False

        if rnn_cell == 'lstm':
            self.rnn = nn.LSTM(self.word_vec_dim, hidden_size, n_layers,
                               batch_first=True, bidirectional=bidirectional,
                               dropout=dropout_p)
        elif rnn_cell == 'gru':
            self.rnn = nn.GRU(self.word_vec_dim, hidden_size, n_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout_p)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell

    def forward(self, input_seq, input_lengths=None):
        embedded = self.embedding(input_seq)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths,
                                                         batch_first=True,
                                                         enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output,
                                                         batch_first=True)
        return output, hidden
