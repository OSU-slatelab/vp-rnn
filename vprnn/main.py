from train import *
from models import *
from utils import *
from collections import defaultdict
from sklearn.metrics import f1_score
import json
import argparse
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import time
import random
import os
import copy
from math import floor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers in BiLSTM')
    parser.add_argument('--attention-unit', type=int, default=350,
                        help='number of attention unit')
    parser.add_argument('--attention-hops', type=int, default=8,
                        help='number of attention hops, for multi-hop attention model')
    parser.add_argument('--pooling', type=str, default='all',
                        help='pooling strategy; choices: [all, mean, max]')
    parser.add_argument('--traindev-data', type=str, default='',
                        help='location of the training data, should be a json file from vp-tokenizer')
    parser.add_argument('--test-data', type=str, default='',
                        help='location of the test data, should be a json file from vp-tokenizer')
    parser.add_argument('--label-data', type=str, default='',
                        help='location of the label map (int -> sentence) in json format')
    parser.add_argument('--dev-ratio', type=float, default=0.1, help='fraction of train-dev data to use for dev')
    parser.add_argument('--kfold', action='store_true', help='use k-fold cross validation')
    parser.add_argument('--kfold-log', type=str, default='rnn.attn.kfold.log',
                        help='location to store kfold logs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--nfc', type=int, default=512,
                        help='hidden (fully connected) layer size for classifier MLP')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--stage2', type=int, default=20,
                        help='number of epochs to run in boosting stage')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='saved_models/rnn_selfatt_model.pt',
                        help='path to save the final model')
    parser.add_argument('--dictionary', type=str, default='dict.json',
                        help='path to save the dictionary, for faster corpus loading')
    parser.add_argument('--word-vector', type=str, default='',
                        help='path for pre-trained word vectors (e.g. GloVe), should be a PyTorch model.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--test-bsize', type=int, default=32,
                        help='batch size for testing')
    parser.add_argument('--nclasses', type=int, default=348,
                        help='number of classes')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='type of optimizer')
    parser.add_argument('--penalization-coeff', type=float, default=0, 
                        help='the attention orthogonality penalization coefficient')

    args = parser.parse_args()
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device = torch.device("cuda")
    
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) # ignored if not --cuda
    random.seed(args.seed)

    # Load Dictionary
    assert os.path.exists(args.traindev_data)
    print('Begin to load the dictionary.')
    dictionary = Dictionary(path=args.dictionary)
    n_token = len(dictionary)

    # Load Data
    print('Begin to load data.')
    traindev_data = open(args.traindev_data).readlines()
    data_test = open(args.test_data).readlines()
    label_data = open(args.label_data).readlines()

    kbest_acc = []
    kbest_f1 = []
    random.shuffle(traindev_data)
    data_folds = split2folds(traindev_data)
    if args.kfold:  # create data folds for k-fold cross-validation
        numf = len(data_folds)
        random.shuffle(traindev_data)
        data_folds = split2folds(traindev_data)
    else: # if testing, split 10% of the data for development
        numf = 1
        data_train,data_val = split_dev(traindev_data, label_data, 0.1)
    flag = True
    for nfold in range(int(numf)): # train for each fold
        if args.kfold:
            if flag:
                len_val, len_test = len(data_folds[nfold//2]) // 2, len(data_folds[nfold//2]) // 2 + len(data_folds[nfold//2]) % 2
                data_val, data_test = data_folds[nfold//2][0:len_val], data_folds[nfold//2][len_val:len_val+len_test]
                data_train = []
                for i,fold in enumerate(data_folds):
                    if i != nfold//2:
                        data_train.extend(fold)
                data_train += label_data
            else:
                data_val, data_test = data_test, data_val
                flag = not flag

        criterion = nn.CrossEntropyLoss()
        model = RnnAtt({'ntoken':n_token,
                     'dictionary':dictionary,
                     'ninp':args.emsize,
                     'word-vector':args.word_vector,
                     'nhid':args.nhid,
                     'nlayers':args.nlayers,
                     'attention-hops':args.attention_hops,
                     'attention-unit':args.attention_unit,
                     'dropout':args.dropout,
                     'nfc':args.nfc,
                     'nclasses':args.nclasses,
                     'pooling':args.pooling
                    })
        model = model.to(device)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
        elif args.optimizer == 'Adadelta':
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
        else:
            raise Exception('For other optimizers, please add it yourself. '
                            'supported ones are: SGD, Adam and Adadelta.')             
        best_f1 = None
        best_model = None
        trainer = Trainer(data_train,dictionary,device,args,criterion=criterion,optimizer=optimizer)
        for epoch in range(args.epochs):
            trainer.data_shuffle()
            train_loss, model = trainer.epoch(epoch+1, model)
            _,acc,f1,_,_ = trainer.evaluate(model, data_val, bsz=args.test_bsize)
            print('-' * 75)
            print(f'| fold {nfold+1} stage 1 epoch {epoch+1} | train loss (total) {train_loss:.8f} | valid f1 {f1:.4f}')
            print('-' * 75)
            if not best_f1 or f1 > best_f1:
                if not args.kfold: save(model, args.save[:-3]+'_'+str(args.seed)+'.best_f1.pt')
                best_f1 = f1
                best_model = copy.deepcopy(model)

        best_boost_f1 = best_f1
        if args.stage2 > 0:
            model = best_model
            model = model.to(device)
            if args.optimizer == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr*0.25, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
            elif args.optimizer == 'Adadelta':
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr*0.25, rho=0.95)
            elif args.optimizer == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
            trainer.update_opt(optimizer)
            for epoch in range(args.stage2):
                trainer.data_shuffle()
                train_loss, model = trainer.epoch(epoch+1, model)
                _,acc,f1,_,_ = trainer.evaluate(model, data_val, bsz=args.test_bsize)
                print('-' * 75)
                print(f'| fold {nfold+1} stage 2 epoch {epoch+1} | train loss (total) {train_loss:.8f} | valid f1 {f1:.4f}')
                print('-' * 75)
                if not best_boost_f1 or f1 > best_boost_f1:
                    if not args.kfold: save(model, args.save[:-3]+'_'+str(args.seed)+'.best_f1.pt')
                    best_boost_f1 = f1
                    best_model = copy.deepcopy(model)

        print(f'| best valid f1. for fold {nfold+1} {best_boost_f1:.4f} |')
        best_model = best_model.to(device)
        _, test_acc, macf1, y_pred, y_true = trainer.evaluate(best_model, data_test, bsz=32)
        print('-' * 75)
        print(f'| test acc. {test_acc:.4f} | test macro. F1 {macf1:.4f} |')
        print('-' * 75)
        if not args.kfold:
            qeval  = QEval(data_train, data_test)
            per_quin_acc, per_quin_f1 = qeval.evaluate(y_pred, y_true) 
            print(f'f1 per quintile = {per_quin_f1}\n')
            print(f'accuracy per quintile = {per_quin_acc}\n')
        kbest_acc.append(test_acc)
        kbest_f1.append(macf1)
        del model, best_model
        torch.cuda.empty_cache()

    if args.kfold:
        with open(args.kfold_log,'a') as f:
            nl = '\n'
            tab = '\t'
            f.write(f'{nl}{nl}{args}{nl}{tab}{kbest_acc}{tab}{np.mean(kbest_acc)}{nl}{tab}{kbest_f1}{tab}{np.mean(kbest_f1)}')
    exit(0)
