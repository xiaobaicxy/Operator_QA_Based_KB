# -*- coding: utf-8 -*-
"""
Bidaf魔改
@author:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证每次运行生成的随机数相同
torch.manual_seed(123)
torch.cuda.manual_seed(123)

class WordEmbedLayer(nn.Module):
    def __init__(self, vocab_size, embed_size, pretrained=None):
        super(WordEmbedLayer, self).__init__()
        if pretrained is not None:
            self.word_embed = nn.Embedding.from_pretrained(pretrained, freeze=True)
        else:
            self.word_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        out = self.word_embed(x) # [batch_size, seq_len, embed_size]
        return out


class ContextualEmbedLayer(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.2):
        super(ContextualEmbedLayer, self).__init__()
        self.bilstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout
                        )

    def forward(self, x, lengths):
        # x: [batch_size, seq_len, embed_size]
        sorted_len, sorted_idx = torch.sort(lengths, descending=True)
        sorted_x = x[sorted_idx.long()]
        _, ori_idx = torch.sort(sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, (h_n, c_n) = self.bilstm(packed_x) 
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = out[ori_idx.long()]
        return out # [batch_size, seq_len, hidden_size*2]


class AttentionFlowLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionFlowLayer, self).__init__()
        self.alpha = nn.Linear(6*hidden_size, 1)
        self.beta = nn.Linear(hidden_size*8, hidden_size*8)

    def forward(self, context, query):
        # context: [batch_size, c_seq_len, hidden_size*2]
        # query: [batch_size, q_seq_len, hidden_size*2]
        batch_size = context.size(0)
        c_seq_len = context.size(1)
        q_seq_len = query.size(1)

        context = context.unsqueeze(2) 
        query = query.unsqueeze(1)
        _context = context.expand(-1, -1, q_seq_len, -1) # [batch_size, c_seq_len, q_seq_len, hidden_size*2]
        _query = query.expand(-1, c_seq_len, -1, -1) # [batch_size, c_seq_len, q_seq_len, hidden_size*2]

        c_q = torch.mul(_context, _query) # [batch_size, c_seq_len, q_seq_len, hidden_size*2],逐元素相乘
        cat1 = torch.cat((_context, _query, c_q), dim=-1)
        S = self.alpha(cat1)
        S = S.squeeze() # [batch_size, c_seq_len, q_seq_len]
        
        query = query.squeeze()
        c_q_atten_w = F.softmax(S, dim=-1)
        query_hat = torch.bmm(c_q_atten_w, query) # [batch_size, c_seq_len, hidden_size*2]

        context = context.squeeze()
        q_c_atten_w = F.softmax(torch.max(S, dim=2)[0], dim=1).unsqueeze(1) # [batch_size, 1, c_seq_len]
        context_hat = torch.bmm(q_c_atten_w, context) # [batch_size, 1, hidden_size*2]
        context_hat = context_hat.expand(-1, c_seq_len, -1) # [batch_size, c_seq_len, hidden_size*2]

        context_query_hat = torch.mul(context, query_hat) # [batch_size, c_seq_len, hidden_size*2]
        context_context_hat = torch.mul(context, context_hat) # [batch_size, c_seq_len, hidden_size*2]
        cat2 = torch.cat((context_hat, query_hat, context_query_hat, context_context_hat), dim=-1)
        out = self.beta(cat2) # [batch_size, c_seq_len, hidden_size*8]

        return out


class OutputLayer(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.2):
        super(OutputLayer, self).__init__()
        self.bilstm = nn.LSTM(input_size=hidden_size*8,
                                    hidden_size=hidden_size,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=dropout
                                )
        self.liner = nn.Linear(hidden_size*2, num_classes)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x, lengths):
        # x: [batch_size, c_seq_len, hidden_size*8]
        sorted_len, sorted_idx = torch.sort(lengths, descending=True)
        sorted_x = x[sorted_idx.long()]
        _, ori_idx = torch.sort(sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_len.long().cpu().data.numpy(), batch_first=True)
        packed_out, (h_n, c_n) = self.bilstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        out = out[ori_idx.long()]

        out = F.avg_pool2d(out, kernel_size=(out.size(1), 1))
        out = out.squeeze(dim=1)  #[batch, hidden_size]
        out = self.liner(out)
        out = self.act_func(out) # [batch_size, num_classes]

        return out 

 

class BidafModel(nn.Module):
    def __init__(self, config):
        super(BidafModel, self).__init__()

        self.word_embed = WordEmbedLayer(config.vocab_size, config.embed_size)
        self.contextual_embed = ContextualEmbedLayer(config.embed_size, config.hidden_size)

        self.att_flow = AttentionFlowLayer(config.hidden_size)
        
        self.output_layer = OutputLayer(config.hidden_size, config.num_classes)

    def forward(self, context, query):
        """
            context = (c_word, c_seq_len)
            query = (q_word, q_seq_len)
            c_word: [batch_size, c_seq_len]
            q_word: [batch_size, q_seq_len]
            c_seq_len, q_seq_len: [batch_size]
        """
        c_word, c_seq_len = context
        q_word, q_seq_len = query
        
        c_embed = self.word_embed(c_word) # [batch_size, c_seq_len, embed_size]
        c_contextual_embed = self.contextual_embed(c_embed, c_seq_len) # [batch_size, c_seq_len, hidden_size*2]

        q_embed = self.word_embed(q_word) # [batch_size, q_seq_len, embed_size]
        q_contextual_embed = self.contextual_embed(q_embed, q_seq_len) # [batch_size, q_seq_len, hidden_size*2]

        G = self.att_flow(c_contextual_embed, q_contextual_embed) # [batch_size, c_seq_len, hidden_size*8]
        out = self.output_layer(G, c_seq_len) # [batch_size, c_seq_len, hidden_size*2]
        
        return out

if __name__ == "__main__":
    from config import Config
    config = Config()
    model = BidafModel(config)