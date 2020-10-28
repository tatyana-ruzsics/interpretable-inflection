import torch
import torch.nn as nn


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, tgt, lengths, **kwargs):
        """
        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
        Returns:
            * decoder output `[tgt_len x batch x hidden]`
            * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        encoder_out = self.encode(src, lengths=lengths, **kwargs)
        self.init_decoder_state(encoder_out["enc_state"])
        dec_out, attns = self.decoder(
            tgt, encoder_out["memory_bank"], memory_lengths=lengths, **kwargs)

        return dec_out, attns

    def encode(self, src, lengths, **kwargs):
        result = dict()
        enc_state, memory_bank = self.encoder(src, lengths=lengths, **kwargs)
        result["enc_state"] = enc_state
        result["memory_bank"] = memory_bank
        return result

    def decode(self, tgt, memory_bank, **kwargs):
        return self.decoder(tgt, memory_bank, **kwargs)

    def init_decoder_state(self, encoder_state):
        self.decoder.init_state(encoder_state)

    def map_decoder_state(self, fn):
        self.decoder.map_state(fn)


class InflectionAttentionModel(NMTModel):

    def __init__(self, encoder, inflection_encoder, decoder, generator):
        super(InflectionAttentionModel, self).__init__(
            encoder, decoder, generator)
        self.inflection_encoder = inflection_encoder

    def forward(self, src, tgt, lengths,
                inflection, inflection_lengths, **kwargs):

        tgt = tgt[:-1]  # exclude last target from inputs

        encoder_out = self.encode(src, lengths, inflection, **kwargs)

        self.init_decoder_state(encoder_out["enc_state"])

        dec_out, attns = self.decode(
            tgt, encoder_out["memory_bank"], memory_lengths=lengths,
            inflection_memory_bank=encoder_out["inflection_memory_bank"],
            inflection_lengths=inflection_lengths, **kwargs)

        return dec_out, attns

    def encode(self, src, lengths, inflection, **kwargs):
        result = dict()
        enc_state, memory_bank = self.encoder(src, lengths=lengths, **kwargs)
        result["enc_state"] = enc_state
        result["memory_bank"] = memory_bank
        inflection_memory_bank = self.inflection_encoder(inflection, **kwargs)
        result["inflection_memory_bank"] = inflection_memory_bank
        return result

    def decode(self, tgt, memory_bank, memory_lengths, inflection_memory_bank,
               inflection_lengths, **kwargs):
        dec_out, attns = self.decoder(
            tgt, memory_bank, memory_lengths=memory_lengths,
            inflection_memory_bank=inflection_memory_bank,
            inflection_lengths=inflection_lengths, **kwargs)
        return dec_out, attns

# iflection attention wiith global gate head
class InflectionGGHAttentionModel(InflectionAttentionModel):

    def __init__(self, encoder, inflection_encoder, global_head, decoder, generator):
        super(InflectionGGHAttentionModel, self).__init__(
            encoder, inflection_encoder, decoder, generator)
        self.global_head = global_head
        #self.fused_msd = fused_msd

    def forward(self, src, tgt, lengths,
                inflection, inflection_lengths, **kwargs):

        tgt = tgt[:-1]  # exclude last target from inputs

        encoder_out = self.encode(src, lengths, inflection, inflection_lengths, **kwargs)

        global_gate_attns = encoder_out['global_gate_attns']

        self.init_decoder_state(encoder_out["enc_state"])

        dec_out, attns = self.decode(
            tgt, encoder_out["memory_bank"], memory_lengths=lengths,
            inflection_memory_bank=encoder_out["inflection_memory_bank"],
            inflection_lengths=inflection_lengths, 
            globalattn_memory_bank=encoder_out['globalattn_memory_bank'], **kwargs)

        return dec_out, {**attns, **global_gate_attns}

    def encode(self, src, lengths, inflection, inflection_lengths, **kwargs):
        result = dict()
        enc_state, memory_bank = self.encoder(src, lengths=lengths, **kwargs)
        result["enc_state"] = enc_state
        result["memory_bank"] = memory_bank
        inflection_memory_bank = self.inflection_encoder(inflection, **kwargs)

        # if self.fused_msd:
        #     fused_msd = torch.mean(inflection_memory_bank, 0, keepdim=True)
        #     #print(inflection_memory_bank.size())
        #     result["inflection_memory_bank"] = torch.cat((inflection_memory_bank,fused_msd),dim=0)
        #     #print(result["inflection_memory_bank"].size())
        # else:
        result["inflection_memory_bank"] = inflection_memory_bank

        # add non-linear layer over enc_state - also try it over a learned vector
        #print(inflection_memory_bank.size())
        pos_memory_bank = inflection_memory_bank[0,:,:].unsqueeze(0)
        #print(pos_memory_bank.unsqueeze(0).size())

        #import pdb; pdb.set_trace()
        if 'word_split' in kwargs:
            # transform memory_bank and memory_lengths to averages within bpe splits
        #    import pdb; pdb.set_trace()
            # make new lengts - list of subwords for each example in batch
            # calc max_len - for padding
            # apply ave within subwords and pad to get max_len

            a = memory_bank
            batch_size=lengths.size()[0]
            split_sizes = kwargs['word_split']
            split_lens = [len(v) for v in split_sizes]
            split_lens2t = torch.tensor(split_lens)
            max_len = max(split_lens)
            #print(max_len)
            # map each src vector to average over hidden vectors within subword spans
            # and pad it to max len of such vectors within a batch
            splits = [torch.cumsum(torch.Tensor([0] + split_sizes[i]), dim=0)[:-1] for i in range(batch_size) ]
            #print(memory_bank.size())
            #print(splits)
            #print(split_sizes)
            b = tuple( torch.nn.ConstantPad1d((0,max_len - split_lens[b_ind]), 0)
                (torch.cat(tuple(torch.mean(a[:,b_ind,:].narrow(0, int(start), int(length)),keepdim=True,dim=0)
                    for start, length in zip(splits[b_ind], split_sizes[b_ind])),dim=0).transpose(0,1)).transpose(0,1).unsqueeze(0)  for b_ind in range(batch_size) )

            memory_bank_subwords = torch.cat(b,dim=0).transpose(0, 1)

            #print(memory_bank_subwords.size())

            globalattn_memory_bank, global_gate_attns = self.global_head(pos_memory_bank,memory_bank_subwords, memory_lengths=split_lens2t,
                                                            inflection_memory_bank=inflection_memory_bank,
                                                            inflection_lengths=inflection_lengths, **kwargs)
        else:

            globalattn_memory_bank, global_gate_attns = self.global_head(pos_memory_bank,memory_bank, memory_lengths=lengths,
                                                            inflection_memory_bank=inflection_memory_bank,
                                                            inflection_lengths=inflection_lengths, **kwargs)
        result['globalattn_memory_bank'] = globalattn_memory_bank
        result['global_gate_attns'] = global_gate_attns
        return result

    def decode(self, tgt, memory_bank, memory_lengths, inflection_memory_bank,
               inflection_lengths, globalattn_memory_bank, **kwargs):
        dec_out, attns = self.decoder(
            tgt, memory_bank, memory_lengths=memory_lengths,
            inflection_memory_bank=inflection_memory_bank,
            inflection_lengths=inflection_lengths, 
            globalattn_memory_bank=globalattn_memory_bank, **kwargs)
        return dec_out, attns


# # iflection attention wiith global gate head
# class InflectionGGHMixedAttentionModel(InflectionAttentionModel):

#     def __init__(self, encoder, inflection_encoder, global_head_subw, global_head_char, decoder, generator):
#         super(InflectionGGHMixedAttentionModel, self).__init__(
#             encoder, inflection_encoder, decoder, generator)
#         self.global_head_subw = global_head_subw
#         self.global_head_char = global_head_char
#         #self.fused_msd = fused_msd

#     def forward(self, src, tgt, lengths,
#                 inflection, inflection_lengths, **kwargs):

#         tgt = tgt[:-1]  # exclude last target from inputs

#         encoder_out = self.encode(src, lengths, inflection, inflection_lengths, **kwargs)

#         global_gate_attns = encoder_out['global_gate_attns']

#         self.init_decoder_state(encoder_out["enc_state"])

#         dec_out, attns = self.decode(
#             tgt, encoder_out["memory_bank"], memory_lengths=lengths,
#             inflection_memory_bank=encoder_out["inflection_memory_bank"],
#             inflection_lengths=inflection_lengths, 
#             globalattn_memory_bank=encoder_out['globalattn_memory_bank'], **kwargs)

#         return dec_out, {**attns, **global_gate_attns}
        
#     def encode(self, src, lengths, inflection, inflection_lengths, **kwargs):
#         result = dict()
#         enc_state, memory_bank = self.encoder(src, lengths=lengths, **kwargs)
#         result["enc_state"] = enc_state
#         result["memory_bank"] = memory_bank
#         inflection_memory_bank = self.inflection_encoder(inflection, **kwargs)

#         # if self.fused_msd:
#         #     fused_msd = torch.mean(inflection_memory_bank, 0, keepdim=True)
#         #     #print(inflection_memory_bank.size())
#         #     result["inflection_memory_bank"] = torch.cat((inflection_memory_bank,fused_msd),dim=0)
#         #     #print(result["inflection_memory_bank"].size())
#         # else:
#         result["inflection_memory_bank"] = inflection_memory_bank

#         # add non-linear layer over enc_state - also try it over a learned vector
#         #print(inflection_memory_bank.size())
#         pos_memory_bank = inflection_memory_bank[0,:,:].unsqueeze(0)

#         # transform memory_bank and memory_lengths to averages within bpe splits
#         # make new lengts - list of subwords for each example in batch
#         # calc max_len - for padding
#         # apply ave within subwords and pad to get max_len

#         globalattn_memory_bank_char, global_gate_attns_char = self.global_head_char(pos_memory_bank,memory_bank, memory_lengths=lengths,
#                                                             inflection_memory_bank=inflection_memory_bank,
#                                                             inflection_lengths=inflection_lengths, **kwargs)

#         a = memory_bank
#         batch_size=lengths.size()[0]
#         split_sizes = kwargs['word_split']
#         split_lens = [len(v) for v in split_sizes]
#         split_lens2t = torch.tensor(split_lens)
#         max_len = max(split_lens)
#         #print(max_len)
#         # map each src vector to average over hidden vectors within subword spans
#         # and pad it to max len of such vectors within a batch
#         splits = [torch.cumsum(torch.Tensor([0] + split_sizes[i]), dim=0)[:-1] for i in range(batch_size) ]
#         #print(memory_bank.size())
#         #print(splits)
#         #print(split_sizes)
#         b = tuple( torch.nn.ConstantPad1d((0,max_len - split_lens[b_ind]), 0)
#             (torch.cat(tuple(torch.mean(a[:,b_ind,:].narrow(0, int(start), int(length)),keepdim=True,dim=0)
#                 for start, length in zip(splits[b_ind], split_sizes[b_ind])),dim=0).transpose(0,1)).transpose(0,1).unsqueeze(0)  for b_ind in range(batch_size) )

#         memory_bank_subwords = torch.cat(b,dim=0).transpose(0, 1)

#         #print(memory_bank_subwords.size())

#         globalattn_memory_bank_suwb, global_gate_attns_subw = self.global_head_subw(pos_memory_bank,memory_bank_subwords, memory_lengths=split_lens2t,
#                                                         inflection_memory_bank=inflection_memory_bank,
#                                                         inflection_lengths=inflection_lengths, **kwargs)

            
#         result['globalattn_memory_bank'] = torch.cat((globalattn_memory_bank_char, globalattn_memory_bank_suwb),2)
#         result['global_gate_attns'] =  {**global_gate_attns_char, **global_gate_attns_subw}
#         return result

#     def decode(self, tgt, memory_bank, memory_lengths, inflection_memory_bank,
#                inflection_lengths, globalattn_memory_bank, **kwargs):
#         dec_out, attns = self.decoder(
#             tgt, memory_bank, memory_lengths=memory_lengths,
#             inflection_memory_bank=inflection_memory_bank,
#             inflection_lengths=inflection_lengths, 
#             globalattn_memory_bank=globalattn_memory_bank, **kwargs)
#         return dec_out, attns

