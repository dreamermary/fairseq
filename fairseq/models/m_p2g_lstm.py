import torch
import torch.nn as nn
from fairseq.models import FairseqEncoder
from fairseq.models import FairseqDecoder
from fairseq.models import FairseqEncoderDecoderModel, register_model

class P2GLSTMEncoder(FairseqEncoder):
    
    def __init__(
        self, args, dictionary, embed_dim=256, hidden_dim=256, dropout=0.1,
    ):
        super().__init__(dictionary)
        self.args = args

        # embeding
        self.embed_tokens = nn.Embedding(
            num_embeddings = len(dictionary),
            embedding_dim = embed_dim,
            padding_idx = dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # maping
        self.lstm = nn.LSTM(
            input_size =embed_dim, # 输入的第三维
            hidden_size = hidden_dim,
            num_layers = 3,
            bidirectional=False, # 需要二维
            batch_first=True,
        )

    def forward(self, src_tokens,src_lengths):
        
        # embedding : [bz,sl] -> [bz,sl,emb_dim]
        x = self.embed_tokens(src_tokens)
        x = self.dropout(x)

        # padding -># 手动padding
        x = nn.utils.rnn.pack_padded_sequence(x, src_lengths,batch_first=True) 

        # mapping
        _outputs, (final_hidden, _final_cell) = self.lstm(x)

        ## ?? [bz, sl * hid_dim]
        return {
            'final_hidden': final_hidden.squeeze(0), ## 应该是1 
        }

    # 编码器输出进行重排序 以 推断使用
    def reorder_encoder_out(self, encoder_out, new_order):
        final_hidden = encoder_out['final_hidden]
        return {
            'final_hidden': final_hidden.index_select(0, new_order)
        }


class P2GLSTMDecoder(FairseqDecoder)
    def __init__(
        self, dictionary, encoder_hidden_dim=256, embed_dim=256, hidden_dim=256, dropout=0.1
    ):
        super().__init__(dictionary)

        # embedding
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)

        # mapping
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_dim + embed_dim, # cat(encoder_out, pre_decoder_out)
            hidden_size=hidden_dim,
            num_layers=3,
            bidirectional=False,
        )

        # out mapping
        self.output_projection = nn.Linear(hidden_dim,len(dictionary))

    def forward(self, prev_output_tokens, encoder_out):
        
        bsz, tgt_len = prev_output_tokens.size()
        final_encoder_hidden = encoder_out['final_hidden] # [bz, encoder_hid_dim]
        
        #embedding
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout(x)
        
        # input -> [bz,trg_len,emb_dim + encoder_hid]
        x = torch.cat( # [bz, tgt_len,emb_dim] + [bz,tgt_len,encoder_dim] # tgt_len = 1
            [x, # [bz, tgt_len,emb_dim]
            final_encoder_hidden.unsqueeze(1).expand(bsz,tgt_len,-1)],  ## final_encoder_hidden: [bz,en_hid] -> [bz,1,en_hid] -> [bz,tgt_len,en_dim] ??
            dim=2
        )
        initial_state = ( # 元组
            final_encoder_hidden.unsqueeze(0), # [1,bz,encoder_hid] # hidden
            torch.zeros_like(final_encoder_hidden).unsqueeze(0), #  #cell
        )

        # mapping : [tgt_len, bz, hid)]
        output, _ = self.lstm(
            x.transpose(0, 1), # [tgt_len, bz, hid]
            initial_state,
        )

        x = output.transpose(0, 1) # [bz, tgt_len,hid]
        x = self.output_projection(x)

        return x, None


@register_model("m_p2g_lstm")
class MP2GLSTMModel(FairseqEncoderDecoderModel):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, metavar=0.1,
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, metavar=0.1,
        )
       
    @classmethod
    def build_model(cls, args, task):
        encoder = P2GLSTMEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim==args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        decoder = P2GLSTMDecoder(
            args=args,
            dictionary=task.target_dictionary,
            embed_dim==args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout,
        )
        model = MP2GLSTMModel(encoder, decoder)
        print(model)

        return model


    # control interact between encoder & decoder : default implementation provided by the FairseqEncoderDecoderModel
    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
    #     return decoder_out




from fairseq.models import register_model_architecture

@register_model_architecture('m_p2g_lstm','my_p2g_lstm')
def my_p2g_lstm(args):

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)









