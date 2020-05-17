# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import os
import torch
import torch.nn as nn
import numpy as np
from .charbilstm import CharBiLSTM
from .charbigru import CharBiGRU
from .charcnn import CharCNN
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer, BertModel


class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        print("build word representation...")
        self.device = torch.device('cuda' if data.HP_gpu and torch.cuda.is_available() else 'cpu')
        self.gpu = data.HP_gpu
        self.pretrain = data.pretrain
        self.bert_model_dir = data.bert_model_dir
        self.elmo_options_file = os.path.join(data.elmo_model_dir, "elmo_2x4096_512_2048cnn_2xhighway_options.json")
        self.elmo_weight_file = os.path.join(data.elmo_model_dir, "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
        self.use_char = data.use_char
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        self.char_all_feature = False
        self.sentence_classification = data.sentence_classification
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_feature_extractor == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == "GRU":
                self.char_feature = CharBiGRU(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_feature_extractor == "ALL":
                self.char_all_feature = True
                self.char_feature = CharCNN(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
                self.char_feature_extra = CharBiLSTM(data.char_alphabet.size(), data.pretrain_char_embedding, self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print("Error char feature selection, please check parameter data.char_feature_extractor (CNN/LSTM/GRU/ALL).")
                exit(0)
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))

        self.feature_num = data.feature_num
        self.feature_embedding_dims = data.feature_emb_dims
        self.feature_embeddings = nn.ModuleList()
        for idx in range(self.feature_num):
            self.feature_embeddings.append(nn.Embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]))
        for idx in range(self.feature_num):
            if data.pretrain_feature_embeddings[idx] is not None:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[idx]))
            else:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(self.random_embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx])))

        if self.pretrain == 'ELMo':
            self.elmo = Elmo(self.elmo_options_file, self.elmo_weight_file, 2, requires_grad=False, dropout=0)
        if self.pretrain == 'BERT': # bert
            self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_dir, do_lower_case=False)
            self.bert_model = BertModel.from_pretrained(self.bert_model_dir)



    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def bert_emb(self, batch_word_list, max_sent_len, select='first'):
        batch_emb = []
        for sent in batch_word_list:
            tokens = []
            slice = []
            tokens.append('[CLS]')
            for word in sent:
                token_set = self.bert_tokenizer.tokenize(word)
                if select == 'first':
                    slice.append(len(tokens))
                else:
                    slice.append(list(range(len(tokens), len(tokens) + len(token_set)))) # [first_idx, ..., last_idx]
                tokens.extend(token_set)
            tokens.append('[SEP]')
            if select == 'first':
                slice += list(range(len(tokens), len(tokens) + max_sent_len - len(sent)))
            else:
                slice += [[idx] for idx in range(len(tokens), len(tokens) + max_sent_len - len(sent))]
            tokens += ['[PAD]'] * (max_sent_len - len(sent))
            assert(len(slice) == max_sent_len)
            input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.tensor([input_ids]).to(self.device)
            with torch.no_grad():
                atten_mask = input_ids.gt(0)
                bert_encode, _ = self.bert_model(input_ids, attention_mask=atten_mask)
                last_hidden = bert_encode[-1]
                if select == 'first':
                    slice = torch.tensor(slice).to(self.device)
                    sent_emb = torch.index_select(last_hidden, 1, slice)
                else:
                    sent_emb = []
                    for slice_token in slice:
                        slice_token = torch.tensor(slice_token).to(self.device)
                        token_emb = torch.mean(torch.index_select(last_hidden, 1, slice_token), dim=1)
                        sent_emb.append(token_emb)
                    sent_emb = torch.cat(sent_emb, dim=0).unsqueeze(0)
                batch_emb.append(sent_emb)

        batch_emb = torch.cat(batch_emb, 0)
        return batch_emb


    def forward(self, original_words_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        """
            input:
                word_inputs: (batch_size, sent_len)
                features: list [(batch_size, sent_len), (batch_len, sent_len),...]
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)

        word_embs = self.word_embedding(word_inputs)

        if self.pretrain == 'ELMo':
            character_ids = batch_to_ids(original_words_batch)
            character_ids = character_ids.to(self.device)
            embeddings = self.elmo(character_ids)
            elmo_embs = embeddings['elmo_representations'][-1]

        if self.pretrain == 'BERT':
            bert_embs = self.bert_emb(original_words_batch, sent_len)


        if self.pretrain == 'ELMo':
            word_list = [torch.cat([elmo_embs, word_embs], -1)]
        elif self.pretrain == 'BERT':
            word_list = [torch.cat([bert_embs, word_embs], -1)]
        else:
            word_list = [word_embs]


        if not self.sentence_classification:
            for idx in range(self.feature_num):
                word_list.append(self.feature_embeddings[idx](feature_inputs[idx]))
        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            ## concat word and char together
            word_list.append(char_features)
            word_embs = torch.cat([word_embs, char_features], 2)
            if self.char_all_feature:
                char_features_extra = self.char_feature_extra.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
                char_features_extra = char_features_extra[char_seq_recover]
                char_features_extra = char_features_extra.view(batch_size,sent_len,-1)
                ## concat word and char together
                word_list.append(char_features_extra)    
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent


