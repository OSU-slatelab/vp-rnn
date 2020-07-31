from models import *

from utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import time
from time import sleep
import sys
import random
import os

from sklearn.metrics import f1_score 

class Trainer:
    def __init__(self, data, dictionary, device, args, criterion = None, optimizer = None):
        self.data = data
        self.dictionary = dictionary
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.args = args 
        self.I = torch.zeros(args.batch_size, args.attention_hops, args.attention_hops)
        for i in range(args.batch_size):
            for j in range(args.attention_hops):
                self.I.data[i][j][j] = 1

    def update_opt(self, optimizer):
        self.optimizer = optimizer

    def data_shuffle(self):
        random.shuffle(self.data)

    def package(self, data, is_train = True):
        data = [json.loads(x) for x in data]
        dat = [[self.dictionary.word2idx[y] for y in x['text']] for x in data]
        maxlen = 0
        for item in dat:
            maxlen = max(maxlen, len(item))
        targets = [x['label'] for x in data]
        maxlen = min(maxlen, 500)
        for i in range(len(data)):
            if maxlen < len(dat[i]):
                dat[i] = dat[i][:maxlen]
            else:
                for j in range(maxlen - len(dat[i])):
                    dat[i].append(self.dictionary.word2idx['<pad>'])
        with torch.set_grad_enabled(is_train):
            dat = torch.tensor(dat, dtype=torch.long)
            targets = torch.tensor(targets, dtype=torch.long)
        return dat.t(), targets        

    def opt_step(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), self.args.clip)
        self.optimizer.step()

    def attn_reg(self, attention, loss):
        I = self.I.to(self.device)
        p_coeff = self.args.penalization_coeff
        attT = torch.transpose(attention, 1, 2).contiguous()
        extra_loss = Frobenius(torch.bmm(attention, attT) - I[:attention.size(0)])
        loss += p_coeff * extra_loss
        return loss

    def evaluate(self, model, data_val, bsz = 32):
        model.eval()
        total_loss = 0
        total_correct = 0
        y_pred = []
        y_true = []
        lbls = []
        for batch, i in enumerate(range(0, len(data_val), bsz)):
            last = min(len(data_val), i+bsz)
            intoks = data_val[i:last]
            data, targets = self.package(intoks, is_train=False)
            data, targets = data.to(self.device), targets.to(self.device)
            hidden = model.init_hidden(data.size(1))
            output, attention = model.forward(data, hidden)
            output_flat = output.view(data.size(1), -1)
            total_loss += self.criterion(output_flat, targets).item()
            prediction = torch.max(output_flat, 1)[1]
            total_correct += torch.sum((prediction == targets).float()).item()
            y_pred.extend(prediction.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
        avg_batch_loss = total_loss / (len(data_val) // bsz)
        acc = total_correct / len(data_val)
        macro_f1 = f1_score(y_true, y_pred, list(set(y_true)), average='macro')
        return avg_batch_loss, acc, macro_f1, y_pred, y_true

    def forward(self, i, model, data, bsz=32, is_train = True):
        last = min(len(data), i+bsz)
        intoks = data[i:last]
        data, targets = self.package(intoks, is_train=is_train)
        data, targets = data.to(self.device), targets.to(self.device) #data --> [seq_len, bsz]
        hidden = model.init_hidden(data.size(1)) #hidden --> [num_dir*num_layer,bsz,nhid]
        output, attention = model.forward(data, hidden)
        loss = self.criterion(output.view(data.size(1), -1), targets)
        if attention is not None:  # add penalization term
            loss = self.attn_reg(attention, loss)
        return loss

    def epoch(self, ep, model):
        model.train()
        total_loss = 0
        for batch, i in enumerate(range(0, len(self.data), self.args.batch_size)):
            loss = self.forward(i, model, self.data, self.args.batch_size, is_train = True)
            self.opt_step(model, loss)
            total_loss += loss.item()
        return total_loss/ ((len(self.data) // self.args.batch_size)), model