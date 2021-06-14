import torch
import torch.nn as nn
from .attention import DecoderAttention


class LSTM_instruction_decoder(nn.Module):
    def __init__(self, output_size, max_len, input_size, hidden_size,
                 n_layers, input_dropout_p=0, dropout_p=0, bidirectional=True,
                 use_attention=True):
        super(LSTM_instruction_decoder, self).__init__()

        # Max prog len
        self.max_len = max_len

        # Output size - not vocab len - linear layers cause we need two outs
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.input_size = input_size
        self.bidirectional_encoder = bidirectional

        # If encoder bidirectional - we need to concatenate
        if self.bidirectional_encoder:
            self.hidden_size *= 2

        self.use_attention = use_attention

        # No embedding: tokens + mask should produce some feature vector

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, n_layers,
                            batch_first=True, dropout=dropout_p)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if use_attention:
            self.attention = DecoderAttention(self.hidden_size)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, y, encoder_outs, encoder_hidden):
        # y is output sequence - we want to predict next feat vec
        # y has to already be concatenation of token + mask
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn_map = self.forward_step(
            y, decoder_hidden, encoder_outs
        )
        return decoder_outputs, decoder_hidden

    def forward_step(self, y, decoder_hidden, encoder_outs):
        batch_size = y.size(0)
        output_size = y.size(1)
        embedded = self.input_dropout(y)

        print(self.hidden_size)
        print(embedded.shape)
        print([h.shape for h in decoder_hidden])
        output, hidden = self.lstm(embedded, decoder_hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outs)

        output = self.out_linear(output.contiguous().view(-1, self.hidden_size))

        return output, hidden, attn


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



