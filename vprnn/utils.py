import json
import math
import pdb
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score 

def save(model, path):
    torch.save(model.state_dict(), path)

def load(model, path):
    model.load_state_dict(torch.load(path))

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True),
                         2, keepdim=True).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def split_dev(all_data, label_data, dev_ratio):
    full_data = label_data + all_data
    label_data_dict = defaultdict(list)
    full_data_js = [json.loads(x) for x in full_data]
    data_val = []
    data_train = []
    for x in full_data_js:
        label_data_dict[x["label"]].append(x["text"])
    for label,texts in label_data_dict.items():
        alen = len(texts) * dev_ratio
        dev_len = math.floor(alen)
        dtexts = texts[:dev_len]
        ttexts = texts[dev_len:]
        data_val.extend([json.dumps({"label":label,"text":txt}) for txt in dtexts])
        data_train.extend([json.dumps({"label":label,"text":txt}) for txt in ttexts])
    return data_train, data_val

def split2folds(data, num_folds = 5):
    data_per_fold = math.floor(1. * len(data) / num_folds)
    folds = []
    for i in range(num_folds):
        folds.append(data[data_per_fold * i:(i+1) * data_per_fold])
    rem_data = data[num_folds * data_per_fold:]
    rem_len = len(rem_data)
    for i in range(rem_len):
        folds[i].append(rem_data[i])
    return folds

class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = list()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class QEval(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.quintiles = self.split_list(self.get_quintiles(), 5)

    def split_list(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def get_quintiles(self):
        full_data_labels = [json.loads(x)["label"] for x in self.train_data]
        lbl_dist = defaultdict(int)
        for lbl in full_data_labels:
            lbl_dist[lbl] += 1
        lbl_dist_tup = [(lbl, freq) for lbl, freq in lbl_dist.items()]
        tdist_sorted = sorted(lbl_dist_tup, key=lambda tup: tup[1])
        lbl_sorted = [x[0] for x in tdist_sorted]
        test_data_labels = [json.loads(x)["label"] for x in self.test_data]
        trim_sorted = [x for x in lbl_sorted if x in test_data_labels]

        return trim_sorted
    
    def evaluate(self, y_pred, y_true):
        evl_acc = []
        evl_f1 = []
        for q in self.quintiles:
            yt,yp=[],[]
            for i,y in enumerate(y_true):
                if y in q:
                    yt.append(y)
                    yp.append(y_pred[i])
            evl_acc.append((np.asarray(yt)==np.asarray(yp)).astype(int).mean())
            evl_f1.append(f1_score(yt,yp,list(set(yt)),average='macro'))
        return evl_acc, evl_f1
