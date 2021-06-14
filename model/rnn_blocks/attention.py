import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, weighted=False):
        super(DecoderAttention, self).__init__()
        self.weighted = weighted
        self.hidden_size = hidden_size
        if self.weighted:
            self.attn_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_output = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, decoder_out, encoder_out):
        # decoder_out BxOxH
        # encoder_out BxIxH
        B, _, H = decoder_out.shape
        assert H == self.hidden_size, "Size mismatch"

        output = decoder_out

        if self.weighted:
            output = self.attn_weight(output)

        attn = torch.bmm(output, encoder_out.transpose(1, 2))   # OxH x HxI
        attn = F.softmax(attn, dim=2)

        context = torch.bmm(attn, encoder_out)
        combined = torch.cat((context, output), dim=2)
        output = torch.tanh(
            self.linear_output(combined)
        )

        return output, attn
