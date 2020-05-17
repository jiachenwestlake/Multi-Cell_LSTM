# -*- coding: utf-8 -*-


from __future__ import print_function
import time
import sys
import math
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

CAT = ['PER', 'ORG', 'LOC', 'MISC']
POSITION = ['I', 'B', 'E', 'S']
LABEL_INDEX = ['O'] + ["{}-{}".format(position, cat) for cat in CAT for position in POSITION]

def data_initialization(data):
    data.initial_feature_alphabets()
    if data.task == 'NER':
        for tag in LABEL_INDEX:
            data.label_alphabet.add(tag)
        data.entity_type.append('O')
        for entity_name in CAT:
            data.entity_type.append(entity_name)
        for entity in data.entity_type:
            data.entity_alphabet.add(entity)
        for pos_name in POSITION:
            data.position_type.append(pos_name)
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.build_alphabet_raw(data.raw_data_dir)
    data.fix_alphabet()
    for i in range(data.label_alphabet.size()-1):
        print(data.label_alphabet.instances[i])
    data.build_entity_dict(data.entity_dict_dir)

def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    if sentence_classification:
        # print(overlaped)
        # print(overlaped*pred)
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0] ## =batch_size
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred)==len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    gold_entity_results = []
    pred_entity_results = []
    gold_probs_results = []
    pred_probs_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        original_words_batch, batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_entity, lm_seq_tensor, mask = batchify_with_label(instance, data.HP_gpu, False, data.sentence_classification)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq, entity_seq, atten_probs_seq = model.decode_nbest(original_words_batch, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq, entity_seq, atten_probs_seq = model(original_words_batch, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_entity, gold_entity = recover_label(entity_seq, batch_entity, mask, data.entity_alphabet, batch_wordrecover, data.sentence_classification)
        pred_entity_results += pred_entity
        gold_entity_results += gold_entity
        pred_probs, gold_probs = recover_label(atten_probs_seq, batch_entity, mask, data.entity_alphabet, batch_wordrecover, data.sentence_classification)
        pred_probs_results += pred_probs
        gold_probs_results += gold_probs

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    print("word acc:")
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    print("entity acc:")
    entity_acc, _, _, _ = get_ner_fmeasure(gold_entity_results, pred_entity_results, "entity predict")
    print("probs acc:")
    probs_acc, _, _, _ = get_ner_fmeasure(gold_probs_results, pred_probs_results, "probs predict")

    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu, if_train=True, sentence_classification=False):
    if sentence_classification:
        return batchify_sentence_classification_with_label(input_batch_list, gpu, if_train)
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train)


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    entities = [sent[4] for sent in input_batch_list]
    original_words = [sent[5] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    entity_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    lm_forward_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    lm_backward_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad = if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).byte()
    for idx, (seq, label, entity, seqlen) in enumerate(zip(words, labels, entities, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        if seqlen > 1:
            lm_forward_seq_tensor[idx, 0: seqlen - 1] = word_seq_tensor[idx, 1: seqlen]
            lm_forward_seq_tensor[idx, seqlen - 1] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 1: seqlen] = word_seq_tensor[idx, 0: seqlen - 1]
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        else:
            lm_forward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
            lm_backward_seq_tensor[idx, 0] = torch.LongTensor([1])  # unk word
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        entity_seq_tensor[idx, :seqlen] = torch.LongTensor(entity)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_lengths = word_seq_lengths.to(device)
    word_seq_tensor = word_seq_tensor[word_perm_idx].to(device)

    # reorder sentence index
    new_original_words = [] # list[list[word]]
    for i in word_perm_idx:
        new_original_words.append(original_words[i])

    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx].to(device)

    lm_forward_seq_tensor = lm_forward_seq_tensor[word_perm_idx].to(device)
    lm_backward_seq_tensor = lm_backward_seq_tensor[word_perm_idx].to(device)
    label_seq_tensor = label_seq_tensor[word_perm_idx].to(device)
    entity_seq_tensor = entity_seq_tensor[word_perm_idx].to(device)
    mask = mask[word_perm_idx].to(device)
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx].to(device)
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    char_seq_recover = char_seq_recover.to(device)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    word_seq_recover = word_seq_recover.to(device)
    lm_seq_tensor = [lm_forward_seq_tensor, lm_backward_seq_tensor]

    return new_original_words, word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, entity_seq_tensor, lm_seq_tensor, mask


def batchify_sentence_classification_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, feature_num), each sentence has one set of feature
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size,), each sentence has one set of feature

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size,), ... ] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, )
            mask: (batch_size, max_sent_len)
    """
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]    
    feature_num = len(features[0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, ), requires_grad = if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad = if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).byte()
    label_seq_tensor = torch.LongTensor(labels)
    # exit(0)
    for idx, (seq,  seqlen) in enumerate(zip(words,  word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_lengths = word_seq_lengths.to(device)
    word_seq_tensor = word_seq_tensor[word_perm_idx].to(device)
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx].to(device)
    label_seq_tensor = label_seq_tensor[word_perm_idx].to(device)
    mask = mask[word_perm_idx].to(device)
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx].to(device)
    char_seq_tensor = char_seq_tensor.to(device)
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    char_seq_recover = char_seq_recover.to(device)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    word_seq_recover = word_seq_recover.to(device)

    return word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask



def train(data):
    print("Training model...")
    device = torch.device('cuda' if torch.cuda.is_available() and data.HP_gpu else 'cpu')
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    if data.sentence_classification:
        model = SentClassifier(data).to(device)
    else:
        model = SeqLabel(data).to(device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    ## compute model parameter num
    n_all_param = sum([p.nelement() for p in model.parameters()])
    n_emb_param = sum([p.nelement() for p in (model.word_hidden.wordrep.word_embedding.weight, model.word_hidden.wordrep.char_feature.char_embeddings.weight, model._LM_softmax.softmax_w,  model._LM_softmax.softmax_b)])
    print("all parameters=%s, emb parameters=%s, other parameters=%s" % (n_all_param, n_emb_param, n_all_param-n_emb_param))
    ## not update the word embedding
    #model.word_hidden.wordrep.word_embedding.weight.requires_grad = False
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)
    best_dev = -10
    test_f = []
    dev_f = []
    best_epoch = 0
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        total_perplexity = 0
        right_entity = 0
        right_atten_probs = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        random.shuffle(data.raw_data_Ids)
        print("Shuffle: first input word list:", data.train_Ids[0][0])
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        raw_data_num = len(data.raw_data_Ids)
        total_batch = train_num//batch_size+1
        raw_batch_size = raw_data_num//total_batch

        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            start_raw = batch_id * raw_batch_size
            end_raw = (batch_id+1) * raw_batch_size
            if end_raw > raw_data_num:
                end_raw = raw_data_num
            instance_raw = data.raw_data_Ids[start_raw:end_raw]
            if not instance:
                continue
            instance_count += 1
            instances = [instance, instance_raw]
            loss_ = 0.0
            for mode_idx, mode in enumerate(['train', 'raw']):
                if len(instance[mode_idx]) < 1:
                    continue
                # print(instance[1])
                original_words_batch, batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_entity, lm_seq_tensor, mask = batchify_with_label(instances[mode_idx], data.HP_gpu, True, data.sentence_classification)
                if mode == 'train':
                    loss, tag_seq, entity_seq, atten_probs_seq = model.calculate_loss(original_words_batch, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, batch_entity, mask)
                    right, whole = predict_check(tag_seq, batch_label, mask, data.sentence_classification)
                    entity_right, entity_whole = predict_check(entity_seq, batch_entity, mask, data.sentence_classification)
                    atten_probs_right, atten_probs_whole = predict_check(atten_probs_seq, batch_entity, mask, data.sentence_classification)
                    right_token += right
                    whole_token += whole
                    right_entity += entity_right
                    right_atten_probs += atten_probs_right
                elif mode == 'raw':
                    loss, perplexity, _,_ = model.raw_loss(original_words_batch, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, batch_entity, lm_seq_tensor, mask)
                    total_perplexity += perplexity.item()
                loss_ += loss

            sample_loss += loss_.item()
            total_loss += loss_.item()

            loss_.backward()
            optimizer.step()
            model.zero_grad()
        LM_perplex = math.exp(total_perplexity / total_batch)
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token, (right_token+0.)/whole_token))
        print("     total perplexity: %.4f" % (LM_perplex))
        print("     entity acc: %.4f"%((right_entity+0.)/whole_token))
        print("     atten probs acc: %.4f" % ((right_atten_probs + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _,_ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))
        dev_f.append(current_score)

        if current_score > best_dev:
            best_epoch = idx
            if data.seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = data.model_dir + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        # ## decode test
        speed, acc, p, r, f, _,_ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            test_f.append(f)
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        else:
            test_f.append(acc)
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
        gc.collect()
        print("The best f in eopch%s, dev:%.4f, test:%.4f" % (best_epoch, dev_f[best_epoch], test_f[best_epoch]))

def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() and data.HP_gpu else 'cpu')
    if data.sentence_classification:
        model = SentClassifier(data).to(device)
    else:
        model = SeqLabel(data).to(device)
    # model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))
    ## compute model parameter num
    n_all_param = sum([p.nelement() for p in model.parameters()])
    n_emb_param = sum([p.nelement() for p in (model.word_hidden.wordrep.word_embedding.weight, model.word_hidden.wordrep.char_feature.char_embeddings.weight, model._LM_softmax.softmax_w,  model._LM_softmax.softmax_b)])
    print("all parameters=%s, emb parameters=%s, other parameters=%s" % (n_all_param, n_emb_param, n_all_param-n_emb_param))
    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--wordemb',  help='Embedding for words', default='None')
    parser.add_argument('--charemb',  help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes") 
    parser.add_argument('--dev', default="data/conll03/dev.bmes" )  
    parser.add_argument('--test', default="data/conll03/test.bmes") 
    parser.add_argument('--seg', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output') 

    args = parser.parse_args()
    data = Data()
    if args.config == 'None':
        data.train_dir = args.train 
        data.dev_dir = args.dev 
        data.test_dir = args.test
        data.model_dir = args.savemodel
        data.dset_dir = args.savedset
        print("Save dset directory:",data.dset_dir)
        save_model_dir = args.savemodel
        data.word_emb_dir = args.wordemb
        data.char_emb_dir = args.charemb
        if args.seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
        print("Seed num:",seed_num)
    else:
        data.read_config(args.config)
    # data.show_data_summary()
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        train(data)
    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        print(data.raw_dir)
        # exit(0)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest and not data.sentence_classification:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

