
from __future__ import print_function
import time
import argparse
import random
import torch
import gc
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data
from utils.data_utils import Multi_Task_Dataset, batch_arrange
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

def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir_S, data.train_dir_T)
    data.build_alphabet(data.dev_dir_S, data.dev_dir_T)
    data.build_alphabet(data.test_dir_S, data.test_dir_T)
    if data.task == 'NER' or data.task == 'Chunk':
        data.NER_create_entity_alphabet()
    elif data.task == 'POSTag':
        data.POS_create_label_alphabet()
    data.fix_alphabet()
    print("label alphabet:")
    print(" ".join([label for label, _ in data.label_alphabet.iteritems()]))


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
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0] ## =batch_size
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
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
    return pred_label, gold_label # list[batch_size*list[seq_len]]


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
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
    lr = init_lr/(1.+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(domain_tag, data, model, name, nbest=None):
    if name == "train":
        if domain_tag == "Source":
            instances = data.train_Ids_S
        elif domain_tag == "Target":
            instances = data.train_Ids_T
    elif name == "dev":
        if domain_tag == "Source":
            instances = data.dev_Ids_S
        elif domain_tag == "Target":
            instances = data.dev_Ids_T
    elif name == 'test':
        if domain_tag == "Source":
            instances = data.test_Ids_S
        elif domain_tag == "Target":
            instances = data.test_Ids_T
    elif name == 'raw':
        if domain_tag == "Target" or domain_tag == "Source":
            instances = data.raw_Ids
    if domain_tag == "Source":
        label_alphabet = data.label_alphabet_S
        entity_alphabet = data.entity_alphabet_S
    elif domain_tag == "Target":
        label_alphabet = data.label_alphabet_T
        entity_alphabet = data.entity_alphabet_T

    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
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
        original_words_batch, batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_entity, mask = batchify_with_label(instance, data.HP_gpu, False, data.sentence_classification)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq, entity_seq, atten_probs_seq = model.decode_nbest(original_words_batch, domain_tag, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest, batch_entity)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq, entity_seq, atten_probs_seq = model(original_words_batch, domain_tag, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # recover entity and probs results
        if entity_seq is not None:
            pred_entity, gold_entity = recover_label(entity_seq, batch_entity, mask, entity_alphabet, batch_wordrecover, data.sentence_classification)
            pred_entity_results += pred_entity
            gold_entity_results += gold_entity
        if atten_probs_seq is not None:
            pred_probs, gold_probs = recover_label(atten_probs_seq, batch_entity, mask, entity_alphabet, batch_wordrecover, data.sentence_classification)
            pred_probs_results += pred_probs
            gold_probs_results += gold_probs

        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    print("word acc:")
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if len(gold_entity_results) > 0:
        print("entity acc")
        entity_acc, _, _, _ = get_ner_fmeasure(gold_entity_results, pred_entity_results, "entity predict")
    if len(gold_probs_results) > 0:
        print("probs acc:")
        probs_acc, _, _, _ = get_ner_fmeasure(gold_probs_results, pred_probs_results, "probs predict")
    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores, pred_entity_results, pred_probs_results
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
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad = if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).byte()
    for idx, (seq, label, entity, seqlen) in enumerate(zip(words, labels, entities, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
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

    label_seq_tensor = label_seq_tensor[word_perm_idx].to(device)
    entity_seq_tensor = entity_seq_tensor[word_perm_idx].to(device)
    mask = mask[word_perm_idx].to(device)
    ### deal with char
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx].to(device)
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    char_seq_recover = char_seq_recover.to(device)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    word_seq_recover = word_seq_recover.to(device)

    return new_original_words, word_seq_tensor, feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, entity_seq_tensor, mask


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
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad = if_train).long()
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
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
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
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    ## compute model parameter num
    n_all_param = sum([p.nelement() for p in model.parameters()])
    n_emb_param = sum([p.nelement() for p in (model.word_hidden.wordrep.word_embedding.weight, model.word_hidden.wordrep.char_feature.char_embeddings.weight)])
    print("all parameters=%s, emb parameters=%s, other parameters=%s" % (n_all_param, n_emb_param, n_all_param-n_emb_param))

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum, weight_decay=data.HP_l2)
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
    train_dataset_S = Multi_Task_Dataset(data.train_Ids_S, data.HP_batch_size)
    train_dataset_T = Multi_Task_Dataset(data.train_Ids_T, data.HP_batch_size)
    total_step = 0
    target_end, source_end = False, False
    epoch_idx = 0
    epoch_start = True # this step is the start of an epoch
    ## start training
    while epoch_idx < data.HP_iteration:
        if epoch_start:
            epoch_start = False
            epoch_loss = 0
            epoch_start_time = time.time()
            print("Epoch: %s/%s" % (epoch_idx, data.HP_iteration))
            if data.optimizer == "SGD":
                optimizer = lr_decay(optimizer, epoch_idx, data.HP_lr_decay, data.HP_lr)
            model.train()
            model.zero_grad()
        if total_step % 2 == 0:
            domain_tag = 'Target'
            batch_instance, target_end = train_dataset_T.next_batch()
        else:
            domain_tag = 'Source'
            batch_instance, source_end = train_dataset_S.next_batch()
        if len(batch_instance) == 0:
            continue
        original_words_batch, batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_entity, mask = \
            batchify_with_label(batch_instance, data.HP_gpu, True, data.sentence_classification)
        loss, entity_loss, atten_probs_loss = model.calculate_loss(original_words_batch, domain_tag, batch_word,
                                                                   batch_features, batch_wordlen, batch_char,
                                                                   batch_charlen, batch_charrecover, batch_label,
                                                                   batch_entity, mask)
        rate = data.HP_target_loss_rate if domain_tag == "Target" else 1.0  # 2:1 for twitter 1.6:1 for bionlp 1.5:1 for broad twitter
        loss_ = rate * loss + entity_loss + atten_probs_loss
        epoch_loss += loss_.item()
        loss_.backward()
        optimizer.step()
        model.zero_grad()
        total_step += 1

        ## evaluation
        if target_end:
            epoch_finish_time = time.time()
            epoch_cost = epoch_finish_time - epoch_start_time
            print("Epoch: %s training finished. Time: %.2fs" % (epoch_idx, epoch_cost))
            print("totalloss:", epoch_loss)
            if epoch_loss > 1e8 or str(epoch_loss) == "nan":
                print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                exit(1)
                continue
            ## decode Target dev
            speed, acc, p, r, f, _,_ = evaluate("Target", data, model, "dev")
            dev_finish_time = time.time()
            dev_cost = dev_finish_time - epoch_finish_time
            if data.seg:
                current_score = f
                print("Dev (Target): time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
            else:
                current_score = acc
                print("Dev (Target): time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))
            dev_f.append(current_score)

            if current_score > best_dev:
                best_epoch = epoch_idx
                if data.seg:
                    print("Exceed previous best f score:", best_dev)
                else:
                    print("Exceed previous best acc score:", best_dev)
                model_name = data.model_dir + ".model"
                print("Save current best model in file:", model_name)
                torch.save(model.state_dict(), model_name)
                best_dev = current_score

            ## decode Target test
            speed, acc, p, r, f, _,_ = evaluate("Target", data, model, "test")
            test_finish_time = time.time()
            test_cost = test_finish_time - dev_finish_time
            if data.seg:
                print("Test (Target): time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
                test_f.append(f)
            else:
                print("Test (Target): time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
                test_f.append(acc)
            gc.collect()
            print("The best f in epoch%s, dev:%.4f, test:%.4f" % (best_epoch, dev_f[best_epoch], test_f[best_epoch]))
            ## epoch end set
            epoch_start = True
            target_end = False
            epoch_idx += 1

        if source_end:
            epoch_finish_time = time.time()
            ## decode test Source
            speed, acc, p, r, f, _,_ = evaluate("Source", data, model, "test")
            test_finish = time.time()
            test_cost = test_finish - epoch_finish_time
            if data.seg:
                print("Test (Source): time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
            else:
                print("Test (Source): time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
            source_end = False

def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() and data.HP_gpu else 'cpu')
    if data.sentence_classification:
        model = SentClassifier(data).to(device)
    else:
        model = SeqLabel(data).to(device)
    ## compute model parameter num
    n_all_param = sum([p.nelement() for p in model.parameters()])
    n_emb_param = sum([p.nelement() for p in (model.word_hidden.wordrep.word_embedding.weight, model.word_hidden.wordrep.char_feature.char_embeddings.weight)])
    print("all parameters=%s, emb parameters=%s, other parameters=%s" % (n_all_param, n_emb_param, n_all_param-n_emb_param))
    model.load_state_dict(torch.load(data.load_model_dir))
    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores, pred_entity_results, pred_prob_results = evaluate("Target", data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores, pred_entity_results, pred_prob_results




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
        print(' '.join(data.entity_type))
        data.read_config(args.config)
        print(data.raw_dir)
        # exit(0)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores, entity_results, prob_results = load_model_decode(data, 'raw')
        if data.nbest and not data.sentence_classification:
            data.write_nbest_decoded_results(decode_results, entity_results, prob_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

