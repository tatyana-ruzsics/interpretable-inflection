import torch
import torch.nn as nn

import onmt.modules
from onmt.modules.output_layer import OutputLayer
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.modules.embedding import MultispaceEmbedding, InflectionLSTMEncoder

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.decoders.decoder import InputFeedRNNDecoder, RNNDecoder
    #InputFeedRNNDecoderGlobalHead
from onmt.modules.global_attention import Attention, TwoHeadedAttention, \
    GatedTwoHeadedAttention, UnsharedGatedTwoHeadedAttention, \
    DecoderUnsharedGatedFourHeadedAttention, GlobalGateTowHeadAttention
    #UnsharedGatedFourHeadedAttention, \
from onmt.modules.sparse_activations import LogSparsemax, Sparsemax
from torchsparseattn import Fusedmax, Oscarmax

from onmt.utils.misc import use_gpu


def load_test_model(opt, dummy_opt, path=None):
    if path is None:
        path = opt.models[0]
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    fields = checkpoint['vocab']
    model_opt = checkpoint['opt']

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_embedding(field, embedding_size):
    pad = field.vocab.stoi[field.pad_token]
    input_size = len(field.vocab)
    return nn.Embedding(input_size, embedding_size, padding_idx=pad)


def build_model(model_opt, fields, gpu, checkpoint=None):
    if model_opt.rnn_size != -1:
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size

    if model_opt.lang_vec_size is not None:
        lang_vec_size = model_opt.lang_vec_size
    else:
        lang_vec_size = model_opt.src_word_vec_size

    if model_opt.infl_vec_size is not None:
        infl_vec_size = model_opt.infl_vec_size
    else:
        infl_vec_size = model_opt.src_word_vec_size

    # Make atomic embedding modules (i.e. not multispace ones)
    lemma_character_embedding = build_embedding(
        fields["src"][0][1], model_opt.src_word_vec_size)
    inflected_character_embedding = build_embedding(
        fields["tgt"][0][1], model_opt.tgt_word_vec_size)
    if "inflection" in fields:
        inflection_embedding = build_embedding(
            fields["inflection"][0][1], infl_vec_size)
    else:
        inflection_embedding = None

    lang_field = fields["language"][0][1] if "language" in fields else None
    lang_embeddings = dict()
    if lang_field is not None and model_opt.lang_location is not None:
        lang_locations = set(model_opt.lang_location)
        if "tgt" in lang_locations:
            assert model_opt.lang_rep != "token", \
                "Can only use a feature for tgt language representation"
        for loc in lang_locations:
            lang_embeddings[loc] = build_embedding(lang_field, lang_vec_size)

    num_langs = len(lang_field.vocab) if "language" in fields else 1

    # Build the full, multispace embeddings
    encoder_embedding = MultispaceEmbedding(
        lemma_character_embedding,
        language=lang_embeddings.get("src", None),
        mode=model_opt.lang_rep
    )

    decoder_embedding = MultispaceEmbedding(
        inflected_character_embedding,
        language=lang_embeddings.get("tgt", None),
        mode=model_opt.lang_rep  # only 'feature' should be allowed here
    )

    if inflection_embedding is not None:
        inflection_embedding = MultispaceEmbedding(
            inflection_embedding,
            language=lang_embeddings.get("inflection", None),
            mode=model_opt.lang_rep
        )
        if model_opt.inflection_rnn:
            if not hasattr(model_opt, "inflection_rnn_layers"):
                model_opt.inflection_rnn_layers = 1
            inflection_embedding = InflectionLSTMEncoder(
                inflection_embedding,
                model_opt.dec_rnn_size,  # need to think about this
                num_layers=model_opt.inflection_rnn_layers,
                dropout=model_opt.dropout,
                bidirectional=model_opt.brnn)
                #, 
                #fused_msd = model_opt.fused_msd)


    # Build encoder
    if model_opt.brnn:
        assert model_opt.enc_rnn_size % 2 == 0
        hidden_size = model_opt.enc_rnn_size // 2
    else:
        hidden_size = model_opt.enc_rnn_size

    enc_rnn = getattr(nn, model_opt.rnn_type)(
            input_size=encoder_embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=model_opt.enc_layers,
            dropout=model_opt.dropout,
            bidirectional=model_opt.brnn)
    encoder = RNNEncoder(encoder_embedding, enc_rnn)

    # Build decoder.
    attn_dim = model_opt.dec_rnn_size
    if model_opt.inflection_attention:
        if model_opt.inflection_gate is not None:
            if model_opt.separate_outputs:
                # if model_opt.separate_heads:
                #     attn = UnsharedGatedFourHeadedAttention.from_options(
                #         attn_dim,
                #         attn_type=model_opt.global_attention,
                #         attn_func=model_opt.global_attention_function,
                #         gate_func=model_opt.inflection_gate
                # )
                # else:
                    attn = UnsharedGatedTwoHeadedAttention.from_options(
                        attn_dim,
                        attn_type=model_opt.global_attention,
                        attn_func=model_opt.global_attention_function,
                        gate_func=model_opt.inflection_gate
                    )
            elif model_opt.global_gate_heads:
                    attn = DecoderUnsharedGatedFourHeadedAttention.from_options(
                        attn_dim,
                        attn_type=model_opt.global_attention,
                        attn_func=model_opt.global_attention_function,
                        gate_func=model_opt.inflection_gate,
                        combine_gate_input=model_opt.global_gate_head_combine,
                        n_global_heads=model_opt.global_gate_heads_number,
                        infl_attn_func=model_opt.infl_attention_function if hasattr(model_opt, 'infl_attention_function') else None
                    )
                    global_head = GlobalGateTowHeadAttention.from_options(
                        attn_dim,
                        attn_type=model_opt.global_attention,
                        attn_func=model_opt.global_attention_function,
                        n_global_heads=model_opt.global_gate_heads_number,
                        tahn_transform=model_opt.global_gate_head_nonlin
                    )
            # elif model_opt.global_gate_heads_mix:
            #         attn = DecoderUnsharedGatedFourHeadedAttention.from_options(
            #             attn_dim,
            #             attn_type=model_opt.global_attention,
            #             attn_func=model_opt.global_attention_function,
            #             gate_func=model_opt.inflection_gate,
            #             combine_gate_input=model_opt.global_gate_head_combine,
            #             n_global_heads=model_opt.global_gate_heads_number * 2,
            #             infl_attn_func=model_opt.infl_attention_function if hasattr(model_opt, 'infl_attention_function') else None
            #         )
            #         global_head_subw = GlobalGateTowHeadAttention.from_options(
            #             attn_dim,
            #             attn_type=model_opt.global_attention,
            #             attn_func=model_opt.global_attention_function,
            #             n_global_heads=model_opt.global_gate_heads_number,
            #             tahn_transform=model_opt.global_gate_head_nonlin,
            #             att_name='gate_lemma_subw_global_'
            #         )
            #         global_head_char = GlobalGateTowHeadAttention.from_options(
            #             attn_dim,
            #             attn_type=model_opt.global_attention,
            #             attn_func=model_opt.global_attention_function,
            #             n_global_heads=model_opt.global_gate_heads_number,
            #             tahn_transform=model_opt.global_gate_head_nonlin,
            #             att_name='gate_lemma_char_global_'
            #         )
            else:
                attn = GatedTwoHeadedAttention.from_options(
                    attn_dim,
                    attn_type=model_opt.global_attention,
                    attn_func=model_opt.global_attention_function,
                    gate_func=model_opt.inflection_gate
                )
        else:
            attn = TwoHeadedAttention.from_options(
                attn_dim,
                attn_type=model_opt.global_attention,
                attn_func=model_opt.global_attention_function
            )
    else:
        attn = Attention.from_options(
            attn_dim,
            attn_type=model_opt.global_attention,
            attn_func=model_opt.global_attention_function
        )

    dec_input_size = decoder_embedding.embedding_dim
    if model_opt.input_feed:
        dec_input_size += model_opt.dec_rnn_size
        stacked_cell = StackedLSTM if model_opt.rnn_type == "LSTM" \
            else StackedGRU
        dec_rnn = stacked_cell(
            model_opt.dec_layers, dec_input_size,
            model_opt.rnn_size, model_opt.dropout
        )
    else:
        dec_rnn = getattr(nn, model_opt.rnn_type)(
            input_size=dec_input_size,
            hidden_size=model_opt.dec_rnn_size,
            num_layers=model_opt.dec_layers,
            dropout=model_opt.dropout)

    #dec_class = InputFeedRNNDecoder if model_opt.input_feed else RNNDecoder
    if model_opt.input_feed:
        decoder = InputFeedRNNDecoder(
            decoder_embedding, dec_rnn, attn, dropout=model_opt.dropout
            )
        # if model_opt.global_gate_heads:
        #     decoder = InputFeedRNNDecoderGlobalHead(
        #         decoder_embedding, dec_rnn, attn, global_head, dropout=model_opt.dropout
        #         )
    else:
        decoder = RNNDecoder(
            decoder_embedding, dec_rnn, attn, dropout=model_opt.dropout
            )
    #decoder = dec_class(
    #    decoder_embedding, dec_rnn, attn, dropout=model_opt.dropout
    #)

    if model_opt.out_bias == 'multi':
        bias_vectors = num_langs
    elif model_opt.out_bias == 'single':
        bias_vectors = 1
    else:
        bias_vectors = 0
    if model_opt.loss == "sparsemax":
        output_transform = LogSparsemax(dim=-1)
        # if model_opt.global_attention_function == "sparsemax":
        #     #output_transform = LogSparsemax(dim=-1)
        #     output_transform = Sparsemax(dim=-1) # adpation, log is applied within OutputLayer class
        # elif model_opt.global_attention_function == "fusedmax":
        #     output_transform = Fusedmax()
        # elif model_opt.global_attention_function == "oscarmax":
        #     output_transform = Oscarmax()
        # else:
        #     print('Unknown global_attention:', model_opt.global_attention)
    else:
        output_transform = nn.LogSoftmax(dim=-1)
    generator = OutputLayer(
        model_opt.dec_rnn_size,
        decoder_embedding.num_embeddings,
        output_transform,
        bias_vectors
    )
    if model_opt.share_decoder_embeddings:
        generator.weight_matrix.weight = decoder.embeddings["main"].weight

    if model_opt.inflection_attention:
        if model_opt.global_gate_heads:
            model = onmt.models.model.InflectionGGHAttentionModel(
            encoder, inflection_embedding, global_head, decoder, generator)
        # elif model_opt.global_gate_heads_mix:
        #     model = onmt.models.model.InflectionGGHMixedAttentionModel(
        #     encoder, inflection_embedding, global_head_subw, global_head_char, decoder, generator)
        else:
            model = onmt.models.model.InflectionAttentionModel(
            encoder, inflection_embedding, decoder, generator)        
    else:
        model = onmt.models.NMTModel(encoder, decoder, generator)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
    elif model_opt.param_init != 0.0:
        for p in model.parameters():
            p.data.uniform_(-model_opt.param_init, model_opt.param_init)

    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    return model
