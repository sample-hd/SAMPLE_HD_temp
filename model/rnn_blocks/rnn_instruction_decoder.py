import torch
import torch.nn as nn
from .attention import DecoderAttention
import numpy as np


class RNN_instruction_decoder(nn.Module):
    def __init__(self, output_size, max_len, input_size, hidden_size,
                 n_layers, input_dropout_p=0, dropout_p=0, rnn_cell='lstm',
                 attention=False):
        super(RNN_instruction_decoder, self).__init__()
        assert rnn_cell in ['lstm', 'gru']

        # Max prog len
        self.max_len = max_len

        # Output size - not vocab len - linear layers cause we need two outs
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.input_size = input_size

        # No embedding: tokens + mask should produce some feature vector
        if rnn_cell == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, n_layers,
                               batch_first=True, dropout=dropout_p)
        elif rnn_cell == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, n_layers,
                              batch_first=True, dropout=dropout_p)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell
        self.attn = attention

    def forward(self, y, hidden_state):
        # y is output sequence - we want to predict next feat vec
        # y has to already be concatenation of token + mask
        batch_size = y.size(0)
        output_size = y.size(1)
        embedded = self.input_dropout(y)

        decoder_outputs, decoder_hidden = self.rnn(embedded, hidden_state)
        # print("decoder_outputs.shape")
        # print(decoder_outputs.shape)

        decoder_outputs = self.out_linear(decoder_outputs)
        # output.view(batch_size, output_size, -1)

        return decoder_outputs, decoder_hidden


class RNN_instruction_decoder_attention(nn.Module):
    def __init__(self, vocab_size, max_len, word_vec_dim, hidden_size,
                 n_layers, input_dropout_p=0, dropout_p=0, rnn_cell='lstm',
                 bidirectional_encoder=True):
        super(RNN_instruction_decoder_attention, self).__init__()
        assert rnn_cell in ['lstm', 'gru']

        # Max prog len
        self.max_len = max_len

        # Output size - not vocab len - linear layers cause we need two outs
        self.output_size = vocab_size
        self.hidden_size = hidden_size

        self.bidirectional_encoder = bidirectional_encoder

        self.word_vec_dim = word_vec_dim

        self.embedding = nn.Embedding(self.output_size, self.word_vec_dim)

        # No embedding: tokens + mask should produce some feature vector
        if rnn_cell == 'lstm':
            self.rnn = nn.LSTM(self.word_vec_dim, self.hidden_size, n_layers,
                               batch_first=True, dropout=dropout_p)
        elif rnn_cell == 'gru':
            self.rnn = nn.GRU(self.word_vec_dim, self.hidden_size, n_layers,
                              batch_first=True, dropout=dropout_p)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        self.hidden_size = hidden_size
        self.rnn_cell = rnn_cell
        self.attention = DecoderAttention(self.hidden_size)

    def forward(self, y, enc_outs, enc_hidden):
        # print("ASDDSFHGFDSH")
        decoder_hidden = self._init_state(enc_hidden)
        decoder_outputs, decoder_hidden, attn_map = self.forward_step(
            y, decoder_hidden, enc_outs
        )
        return decoder_outputs, decoder_hidden

    def forward_step(self, inp, hidden, enc_outs):
        # y is output sequence - we want to predict next feat vec
        # y has to already be concatenation of token + mask

        # print(inp)
        embedded = self.embedding(inp)
        embedded = self.input_dropout(embedded)

        decoder_outputs, decoder_hidden = self.rnn(embedded, hidden)

        output, attn = self.attention(decoder_outputs, enc_outs)
        # print("decoder_outputs.shape")
        # print(decoder_outputs.shape)
        output = self.out_linear(output)
        # output.view(batch_size, output_size, -1)

        return output, decoder_hidden, attn

    def sample(self, start_tok, enc_outs, enc_hidden, end_tok):
        # print(start_tok.shape)
        decoder_hidden = self._init_state(enc_hidden)
        output_symbols = []
        output_lengths = np.array([self.max_len] * start_tok.shape[0])
        decoder_input = start_tok

        def decode(i, out):
            # print(out)
            symbols = out.topk(1)[1]
            output_symbols.append(symbols.squeeze())
            # print(symbols)
            eos = symbols.data.eq(end_tok)
            if eos.dim() > 0:
                eos = eos.cpu().view(-1).numpy()
                # Mask places where end symbol appeared and output_length
                # was never updated for length update
                update_idx = ((output_lengths > i) & eos) != 0
                # Address by bools and update with current prog length
                output_lengths[update_idx] = len(output_symbols)
            symbols = symbols.squeeze(2)
            # print(symbols.shape)
            return symbols

        for i in range(self.max_len):
            decoder_outputs, decoder_hidden, attn_map = self.forward_step(
                decoder_input, decoder_hidden, enc_outs
            )
            decoder_input = decode(i, decoder_outputs)

        output_symbols = torch.stack(output_symbols, dim=1)
        # print(output_symbols.shape)
        # print(output_symbols)
        return output_symbols

    # From seq2seq model:
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    # From seq2seq model:
    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h
