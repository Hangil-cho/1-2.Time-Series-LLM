class Decodcer(nn.Module):
    def __init__(self):
        super().__init__()

        self.selfattn = MultiHeadAttention()
        self.layernorm_1 = LayerNorm()
        self.feedforward = Poswise_FeedForward()
        self.layernorm_2 = LayerNorm()

    def forward(self, decoder_input, masking):
        output, attn_prob = self.selfattn(decoder_input, decoder_input, decoder_input, masking)
        output1 = self.layernorm_1(output + decoder_input)
        output2 = self.feedforward(output1)
        output3 = self.layernorm_2(output2 + output)

        return output3, attn_prob
