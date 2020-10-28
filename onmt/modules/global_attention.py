import torch
import torch.nn as nn

from onmt.modules.sparse_activations import Sparsemax
from torchsparseattn import Fusedmax, Oscarmax
from onmt.utils.misc import aeq


def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths."""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class DotScorer(nn.Module):
    def __init__(self):
        super(DotScorer, self).__init__()

    def forward(self, h_t, h_s):
        return torch.bmm(h_t, h_s.transpose(1, 2))


class GeneralScorer(nn.Module):
    def __init__(self, dim):
        super(GeneralScorer, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)

    def forward(self, h_t, h_s):
        h_t = self.linear_in(h_t)
        return torch.bmm(h_t, h_s.transpose(1, 2))


class AttentionHead(nn.Module):
    def __init__(self, score, transform):
        super(AttentionHead, self).__init__()
        self.score = score
        self.transform = transform

    @classmethod
    def from_options(cls, dim, attn_type="dot", attn_func="softmax"):
        str2score = {
            "dot": DotScorer(),
            "general": GeneralScorer(dim)
        }
        str2func = {
            "softmax": nn.Softmax(dim=-1),
            "sparsemax": Sparsemax(dim=-1),
            "fusedmax": Fusedmax(),
            "oscarmax": Oscarmax()
        }
        score = str2score[attn_type]
        transform = str2func[attn_func]

        return cls(score, transform)

    def forward(self, query, memory_bank, memory_lengths=None, **kwargs):
        """
        query (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
        memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
        memory_lengths (`LongTensor`): the source context lengths `[batch]`

        returns attention distribution (tgt_len x batch x src_len)
        """
        src_batch, src_len, src_dim = memory_bank.size()
        tgt_batch, tgt_len, tgt_dim = query.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)

        align = self.score(query, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            align.masked_fill_(~mask.unsqueeze(1), -float('inf'))

        #import pdb; pdb.set_trace()
        # it should not be necessary to view align as a 2d tensor, but
        # something is broken with sparsemax and it cannot handle a 3d tensor
        #print(align.size())
        #print(align.view(-1, src_len).size())

        #print(src_len)
        #print(src_batch)
        #return self.transform(align.view(-1, src_len), lengths=torch.tensor([src_len]*src_batch)).view_as(align)
        #return self.transform(align.view(-1, src_len), lengths=memory_lengths).view_as(align)
        return self.transform(align.view(-1, src_len)).view_as(align)


class Attention(nn.Module):
    def __init__(self, attention_head, output_layer):
        super(Attention, self).__init__()
        self.attention_head = attention_head
        self.output_layer = output_layer

    @classmethod
    def from_options(cls, dim, attn_type="dot", attn_func="softmax"):
        attention_head = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        # not gonna worry about Bahdanau attention...
        output_layer = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        return cls(attention_head, output_layer)

    def forward(self, query, memory_bank, memory_lengths=None, **kwargs):
        """
        query (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
        memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
        memory_lengths (`LongTensor`): the source context lengths `[batch]`

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        memory_bank = memory_bank.transpose(0, 1)

        align_vectors = self.attention_head(query, memory_bank, memory_lengths)

        # each context vector c_t is the weighted average over source states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate (this step is different in the multiheaded case)
        concat_c = torch.cat([c, query], 2)
        attn_h = self.output_layer(concat_c)

        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()

        return attn_h, {"lemma": align_vectors}


class TwoHeadedAttention(nn.Module):

    def __init__(self, lemma_attn, inflection_attn, output_layer):
        super(TwoHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.output_layer = output_layer

    @classmethod
    def from_options(cls, dim, attn_type="dot", attn_func="softmax"):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        attn_output_layer = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False), nn.Tanh()
        )
        return cls(lemma_attn, inflection_attn, attn_output_layer)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)
        concat_context = torch.cat([lemma_context, infl_context, query], 2)

        attn_h = self.output_layer(concat_context)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        return attn_h, {"lemma": lemma_align, "inflection": infl_align}


class GatedTwoHeadedAttention(nn.Module):

    def __init__(self, lemma_attn, inflection_attn, output_layer, gate):
        super(GatedTwoHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.output_layer = output_layer
        self.gate = gate

    @classmethod
    def from_options(cls, dim, attn_type="dot",
                     attn_func="softmax", gate_func="softmax"):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        attn_output_layer = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        str2func = {
            "softmax": nn.Softmax(dim=-1), "sparsemax": Sparsemax(dim=-1)
        }
        gate_transform = str2func[gate_func]

        # try it with bias?
        gate = nn.Sequential(nn.Linear(dim * 3, 2, bias=True), gate_transform)
        return cls(lemma_attn, inflection_attn, attn_output_layer, gate)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        batch_size, tgt_len, rnn_size = query.size()

        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)

        concat_context = torch.cat([lemma_context, infl_context, query], 2)
        concat_context = concat_context.view(batch_size * tgt_len, -1)
        gate_vec = self.gate(concat_context).view(batch_size, tgt_len, -1, 1)

        lemma_attn_h = self.output_layer(torch.cat([query, lemma_context], 2))
        infl_attn_h = self.output_layer(torch.cat([query, infl_context], 2))
        stacked_h = torch.stack([lemma_attn_h, infl_attn_h], dim=2)
        attn_h = (gate_vec * stacked_h).sum(2)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        gate = gate_vec.transpose(0, 1).squeeze(3).contiguous()

        attns = {"lemma": lemma_align, "inflection": infl_align, "gate": gate}
        return attn_h, attns


class UnsharedGatedTwoHeadedAttention(nn.Module):
    """
    This, like many of the attention mechanisms described, can probably be
    refactored away. But not until after the SIGMORPHON submission.
    """

    def __init__(self, lemma_attn, inflection_attn,
                 lemma_output_layer, inflection_output_layer, gate):
        super(UnsharedGatedTwoHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.lemma_out = lemma_output_layer
        self.infl_out = inflection_output_layer
        self.gate = gate

    @classmethod
    def from_options(cls, dim, attn_type="dot",
                     attn_func="softmax", gate_func="softmax"):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        lemma_out = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        infl_out = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        str2func = {
            "softmax": nn.Softmax(dim=-1), "sparsemax": Sparsemax(dim=-1)
        }
        gate_transform = str2func[gate_func]

        # try it with bias?
        gate = nn.Sequential(nn.Linear(dim * 3, 2, bias=True), gate_transform)
        return cls(lemma_attn, inflection_attn, lemma_out, infl_out, gate)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        batch_size, tgt_len, rnn_size = query.size()

        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)

        concat_context = torch.cat([lemma_context, infl_context, query], 2)
        concat_context = concat_context.view(batch_size * tgt_len, -1)
        gate_vec = self.gate(concat_context).view(batch_size, tgt_len, -1, 1)

        lemma_attn_h = self.lemma_out(torch.cat([query, lemma_context], 2))
        infl_attn_h = self.infl_out(torch.cat([query, infl_context], 2))
        stacked_h = torch.stack([lemma_attn_h, infl_attn_h], dim=2)
        attn_h = (gate_vec * stacked_h).sum(2)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        gate = gate_vec.transpose(0, 1).squeeze(3).contiguous()

        attns = {"lemma": lemma_align, "inflection": infl_align, "gate": gate}
        return attn_h, attns

# class UnsharedGatedFourHeadedAttention(nn.Module):
#     """
#     Replace attention for gate with separate lemma attention and average of msd
#     """

#     def __init__(self, lemma_attn, inflection_attn,
#                  lemma_output_layer, inflection_output_layer, gate, gate_lemma_attn, gate_inflection_attn):
#         super(UnsharedGatedFourHeadedAttention, self).__init__()
#         self.lemma_attn = lemma_attn
#         self.inflection_attn = inflection_attn
#         self.lemma_out = lemma_output_layer
#         self.infl_out = inflection_output_layer
#         self.gate = gate
#         self.gate_lemma_attn = gate_lemma_attn
#         self.gate_inflection_attn = gate_inflection_attn

#     @classmethod
#     def from_options(cls, dim, attn_type="dot",
#                      attn_func="softmax", gate_func="softmax"):
#         lemma_attn = AttentionHead.from_options(
#             dim, attn_type=attn_type, attn_func=attn_func)
#         inflection_attn = AttentionHead.from_options(
#             dim, attn_type=attn_type, attn_func=attn_func)
#         lemma_out = nn.Sequential(
#             nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
#         )
#         infl_out = nn.Sequential(
#             nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
#         )
#         str2func = {
#             "softmax": nn.Softmax(dim=-1), "sparsemax": Sparsemax(dim=-1)
#         }
#         gate_transform = str2func[gate_func]

#         # try it with bias?
#         gate = nn.Sequential(nn.Linear(dim * 3, 2, bias=True), gate_transform)

#         gate_lemma_attn = AttentionHead.from_options(
#             dim, attn_type=attn_type, attn_func=attn_func)
#         gate_inflection_attn = AttentionHead.from_options(
#             dim, attn_type=attn_type, attn_func=attn_func)

#         return cls(lemma_attn, inflection_attn, lemma_out, infl_out, gate, gate_lemma_attn, gate_inflection_attn)

#     def forward(self, query, memory_bank, inflection_memory_bank,
#                 memory_lengths=None, inflection_lengths=None, **kwargs):
#         """
#         query: FloatTensor, (batch size x target len x rnn size)
#         memory_bank: FloatTensor, (source len x batch size x rnn size)
#         inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
#         *lengths: LongTensor, (batch_size)

#         returns attentional hidden state (tgt_len x batch x dim) and
#         attention distribution (tgt_len x batch x src_len)
#         """
#         batch_size, tgt_len, rnn_size = query.size()

#         memory_bank = memory_bank.transpose(0, 1)
#         inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

#         gate_lemma_align = self.gate_lemma_attn(query, memory_bank, memory_lengths)
#         gate_infl_align = self.gate_inflection_attn(
#             query, inflection_memory_bank, inflection_lengths)
#         gate_lemma_context = torch.bmm(gate_lemma_align, memory_bank)
#         gate_infl_context = torch.bmm(gate_infl_align, inflection_memory_bank)

#         gate_concat_context = torch.cat([gate_lemma_context, gate_infl_context, query], 2)
#         gate_concat_context = gate_concat_context.view(batch_size * tgt_len, -1)
#         gate_vec = self.gate(gate_concat_context).view(batch_size, tgt_len, -1, 1)

#         # state vectors

#         lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
#         infl_align = self.inflection_attn(
#             query, inflection_memory_bank, inflection_lengths)
#         lemma_context = torch.bmm(lemma_align, memory_bank)
#         infl_context = torch.bmm(infl_align, inflection_memory_bank)

#         concat_context = torch.cat([lemma_context, infl_context, query], 2)
#         concat_context = concat_context.view(batch_size * tgt_len, -1)

#         lemma_attn_h = self.lemma_out(torch.cat([query, lemma_context], 2))
#         infl_attn_h = self.infl_out(torch.cat([query, infl_context], 2))
#         stacked_h = torch.stack([lemma_attn_h, infl_attn_h], dim=2)
#         attn_h = (gate_vec * stacked_h).sum(2)

#         attn_h = attn_h.transpose(0, 1).contiguous()
#         lemma_align = lemma_align.transpose(0, 1).contiguous()
#         infl_align = infl_align.transpose(0, 1).contiguous()
#         gate = gate_vec.transpose(0, 1).squeeze(3).contiguous()

#         attns = {"lemma": lemma_align, "inflection": infl_align, "gate": gate, 
#                     'gate_lemma_align': gate_lemma_align, 'gate_infl_align': gate_infl_align}
#         return attn_h, attns

class GlobalGateTowHeadAttention(nn.Module):
    """
    Replace attention for gate with separate lemma attention and average of msd
    """

    def __init__(self, lemma_attn, tahn_transform, tahn_layer, n_global_heads, att_name):
        super(GlobalGateTowHeadAttention, self).__init__()

        #self.lemma_attn = lemma_attn

        self.lemma_attn = nn.ModuleList([lemma_attn for i in range(n_global_heads)])
        #self.inflection_attn = inflection_attn
        self.tahn_transform = tahn_transform
        self.tahn_layer = nn.ModuleList([tahn_layer for i in range(n_global_heads)])
        self.n_global_heads = n_global_heads
        self.att_name = att_name

    @classmethod
    def from_options(cls, dim, attn_type="dot",
                     attn_func="softmax", n_global_heads = 1, 
                     tahn_transform=False, att_name='gate_lemma_global_'):

        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        #lemma_attn = []
        #for _ in range(n_global_heads):
        #    lemma_attn.append(AttentionHead.from_options(
        #    dim, attn_type=attn_type, attn_func=attn_func))
        #lemma_attn = [AttentionHead.from_options(
        #    dim, attn_type=attn_type, attn_func=attn_func) for _ in range(n_global_heads)]
        #inflection_attn = AttentionHead.from_options(
        #    dim, attn_type=attn_type, attn_func=attn_func)
        if tahn_transform:
            tahn_layer = nn.Sequential(
                nn.Linear(dim, dim, bias=False), nn.Tanh()
                )
            #tahn_layer = []
            #for _ in range(n_global_heads):
            #    tahn_layer.append(nn.Sequential(
            #    nn.Linear(dim, dim, bias=False), nn.Tanh()
            #    ))
        else:
            tahn_layer=None

        #return cls(lemma_attn, inflection_attn)
        return cls(lemma_attn, tahn_transform, tahn_layer, n_global_heads, att_name)

    def forward(self, query, memory_bank, inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (1 x batch size x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (1 x batch x dim) and
        attention distribution (1 x batch x src_len)
        """

        memory_bank = memory_bank.transpose(0, 1)

        query = query.transpose(0, 1)


        lemma_context = []
        global_attns = {}


        for i,(l,t) in enumerate(zip(self.lemma_attn,self.tahn_layer)):

            if self.tahn_transform:
                query = t(query)
            lemma_align = l(query, memory_bank, memory_lengths)
            lemma_context.append(torch.bmm(lemma_align, memory_bank))

            lemma_align = lemma_align.transpose(0, 1).contiguous()
            global_attn_key =  self.att_name + str(i)
            global_attns[global_attn_key] = lemma_align

        lemma_context = torch.cat(lemma_context, 2)
        #print(lemma_context.size())

        return lemma_context, global_attns

class DecoderUnsharedGatedFourHeadedAttention(nn.Module):
    """
    Replace attention for gate with separate lemma attention and average of msd
    """

    def __init__(self, lemma_attn, inflection_attn,
                 lemma_output_layer, inflection_output_layer, gate, combine_gate_input):
        super(DecoderUnsharedGatedFourHeadedAttention, self).__init__()
        self.lemma_attn = lemma_attn
        self.inflection_attn = inflection_attn
        self.lemma_out = lemma_output_layer
        self.infl_out = inflection_output_layer
        self.gate = gate
        self.combine_gate_input = combine_gate_input

    @classmethod
    def from_options(cls, dim, attn_type="dot",
                     attn_func="softmax", gate_func="softmax", 
                     combine_gate_input=False, n_global_heads = 1, infl_attn_func=None):
        lemma_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=attn_func)
        if infl_attn_func==None: 
            infl_attn_func=attn_func
        inflection_attn = AttentionHead.from_options(
            dim, attn_type=attn_type, attn_func=infl_attn_func)
        lemma_out = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        infl_out = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False), nn.Tanh()
        )
        str2func = {
            "softmax": nn.Softmax(dim=-1), "sparsemax": Sparsemax(dim=-1)
        }
        gate_transform = str2func[gate_func]

        # try it with bias?
        if combine_gate_input:
            # input is global head (1 or more), two local heads and query (decoder state)
            gate = nn.Sequential(nn.Linear(dim * (n_global_heads+ 3), 2, bias=True), gate_transform)
        else:
            # input is global head (1 or more) and query (decoder state)
            gate = nn.Sequential(nn.Linear(dim * (n_global_heads + 1), 2, bias=True), gate_transform)


        return cls(lemma_attn, inflection_attn, lemma_out, infl_out, gate, combine_gate_input)

    def forward(self, query, memory_bank, globalattn_memory_bank, 
                inflection_memory_bank,
                memory_lengths=None, inflection_lengths=None, **kwargs):
        """
        query: FloatTensor, (batch size x target len x rnn size)
        memory_bank: FloatTensor, (source len x batch size x rnn size)
        inflection_memory_bank: FloatTensor, (infl len x batch size x rnn size)
        globalattn_memory_bank: List of FloatTensor (batch size x rnn size)
        *lengths: LongTensor, (batch_size)

        returns attentional hidden state (tgt_len x batch x dim) and
        attention distribution (tgt_len x batch x src_len)
        """
        batch_size, tgt_len, rnn_size = query.size()

        # print(globalattn_memory_bank[0].size())
        # print(globalattn_memory_bank[1].size())
        #print(globalattn_memory_bank.size())
        #print(query.size())

        #lemma_context, infl_context = globalattn_memory_bank
        #gate_concat_context = torch.cat([lemma_context, infl_context, query], 2)
        # gate_concat_context = torch.cat([globalattn_memory_bank, query], 2)
        # #print(globalattn_memory_bank.size())
        # #print(query.size())
        # #print(gate_concat_context.size())
        # gate_concat_context = gate_concat_context.view(batch_size * tgt_len, -1)
        # gate_vec = self.gate(gate_concat_context).view(batch_size, tgt_len, -1, 1)

        # state vectors

        # print('query', query.size())
        # print('memory_bank', memory_bank.size())

        memory_bank = memory_bank.transpose(0, 1)
        inflection_memory_bank = inflection_memory_bank.transpose(0, 1)

        lemma_align = self.lemma_attn(query, memory_bank, memory_lengths)
        infl_align = self.inflection_attn(
            query, inflection_memory_bank, inflection_lengths)
        lemma_context = torch.bmm(lemma_align, memory_bank)
        infl_context = torch.bmm(infl_align, inflection_memory_bank)


        if self.combine_gate_input:
            #gate_concat_context = torch.cat([*globalattn_memory_bank, lemma_context, infl_context, query], 2)
            gate_concat_context = torch.cat([globalattn_memory_bank, lemma_context, infl_context, query], 2)
            gate_concat_context = gate_concat_context.view(batch_size * tgt_len, -1)
            gate_vec = self.gate(gate_concat_context).view(batch_size, tgt_len, -1, 1)
        else:
            #gate_concat_context = torch.cat([*globalattn_memory_bank, query], 2)
            gate_concat_context = torch.cat([globalattn_memory_bank, query], 2)
            gate_concat_context = gate_concat_context.view(batch_size * tgt_len, -1)
            gate_vec = self.gate(gate_concat_context).view(batch_size, tgt_len, -1, 1)

        concat_context = torch.cat([lemma_context, infl_context, query], 2)
        concat_context = concat_context.view(batch_size * tgt_len, -1)

        lemma_attn_h = self.lemma_out(torch.cat([query, lemma_context], 2))
        infl_attn_h = self.infl_out(torch.cat([query, infl_context], 2))
        stacked_h = torch.stack([lemma_attn_h, infl_attn_h], dim=2)
        attn_h = (gate_vec * stacked_h).sum(2)

        attn_h = attn_h.transpose(0, 1).contiguous()
        lemma_align = lemma_align.transpose(0, 1).contiguous()
        infl_align = infl_align.transpose(0, 1).contiguous()
        gate = gate_vec.transpose(0, 1).squeeze(3).contiguous()

        attns = {"lemma": lemma_align, "inflection": infl_align, "gate": gate}
        return attn_h, attns