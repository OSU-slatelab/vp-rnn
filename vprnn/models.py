import pdb
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class Embed(nn.Module):
    def __init__(self,ntoken, dictionary, ninp, word_vector=None):
        super(Embed, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.dictionary = dictionary
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(word_vector):
            print('Loading word vectors from', word_vector)
            vectors = torch.load(word_vector)
            assert vectors[3] >= ninp
            vocab = vectors[1]
            vectors = vectors[2]
            loaded_cnt = 0
            unseen_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    to_add = torch.zeros_like(vectors[0]).uniform_(-0.25,0.25)
                    print("uncached word: " + word)
                    unseen_cnt += 1
                    #print(to_add)
                else:
                    loaded_id = vocab[word]
                    to_add = vectors[loaded_id][:ninp]
                    loaded_cnt += 1
                real_id = self.dictionary.word2idx[word]
                self.encoder.weight.data[real_id] = to_add
            print('%d words from external word vectors loaded, %d unseen' % (loaded_cnt, unseen_cnt))  
      
    def forward(self,input):
        return self.encoder(input)

class RNN(nn.Module):
    def __init__(self, inp_size, nhid, nlayers):
        super(RNN, self).__init__()
        self.nlayers = nlayers
        self.nhid = nhid
        self.rnn = nn.GRU(inp_size, nhid, nlayers, bidirectional=True)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return torch.zeros(self.nlayers * 2, bsz, self.nhid, dtype=weight.dtype,
                            layout=weight.layout, device=weight.device)

    def forward(self, input, hidden):
        out_rnn = self.rnn(input, hidden)[0]
        return out_rnn

class Attention(nn.Module):
    def __init__(self,inp_size, attention_unit, attention_hops, dictionary, dropout):
        super(Attention,self).__init__()
        self.ws1 = nn.Linear(inp_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.dictionary = dictionary
        self.attention_hops = attention_hops
        self.drop = nn.Dropout(dropout)

    def get_mask(self,input_raw):
        transformed_inp = torch.transpose(input_raw, 0, 1).contiguous()  # [bsz, seq_len]
        transformed_inp = transformed_inp.view(input_raw.size()[1], 1, input_raw.size()[0])  # [bsz, 1, seq_len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, seq_len]
        mask = (concatenated_inp == self.dictionary.word2idx['<pad>']).float()
        mask = mask[:,:,:input_raw.size(0)]
        return mask

    def forward(self, input, input_raw): # input --> (seq_len, bsize, inp_size) input_raw --> (seq_len, bsize)
        inp = torch.transpose(input, 0, 1).contiguous()
        size = inp.size()  # [bsz, seq_len, inp_size]
        compressed_embeddings = inp.view(-1, size[2])  # [bsz*seq_len, inp_size]
        mask = self.get_mask(input_raw) # need this to mask out the <pad>s
        hbar = torch.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*seq_len, attention-unit]
        alphas = self.ws2(self.drop(hbar)).view(size[0], size[1], -1)  # [bsz, seq_len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, seq_len]
        penalized_alphas = alphas + -10000*mask
        alphas = F.softmax(penalized_alphas.view(-1, size[1]),1)  # [bsz*hop, seq_len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, seq_len]
        out_agg, attention = torch.bmm(alphas, inp), alphas # [bsz, hop, inp_size], [bsz, hop, seq_len] 
        return out_agg, attention

class Classifier(nn.Module):
    def __init__(self,inp_size, nfc, nclasses, dropout):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(inp_size, nfc)
        self.pred = nn.Linear(nfc, nclasses)
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        fc = torch.tanh(self.fc(self.drop(input)))
        pred = self.pred(self.drop(fc))
        return pred

class RnnAtt(nn.Module):
    def __init__(self,config):
        super(RnnAtt, self).__init__()
        self.emb = Embed(config['ntoken'], config['dictionary'], config['ninp'], config['word-vector'])
        self.rnn = RNN(config['ninp'], config['nhid'], config['nlayers'])
        self.attention = Attention(2*config['nhid'], config['attention-unit'], config['attention-hops'], config['dictionary'], config['dropout'])
        self.classifier = Classifier(2*config['nhid'] * config['attention-hops'], config['nfc'], config['nclasses'], config['dropout'])
        self.drop = nn.Dropout(config['dropout'])
        self.pooling = config['pooling']
        self.hops = config['attention-hops']

    def init_hidden(self,bsz):
        return self.rnn.init_hidden(bsz)

    def forward(self,input,hidden):
        emb_out = self.emb(input)
        rnn_out = self.rnn(self.drop(emb_out),hidden)
        '''
        if self.pooling == 'mean':
            out_agg, attention = torch.mean(rnn_out, 0).squeeze(),None
            out_agg = torch.unsqueeze(out_agg,1)
            out_agg = out_agg.repeat(1,self.hops,1)
        '''
        out_agg, attn = self.attention(rnn_out,input)
        out_agg = out_agg.view(out_agg.size(0), -1)
        pred = self.classifier(out_agg)
        return pred, attn

    def flatten_parameters(self):
        self.rnn.rnn.flatten_parameters()
