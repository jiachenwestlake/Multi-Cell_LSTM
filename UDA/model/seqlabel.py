# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF
from .sampled_softmax_loss import SampledSoftmaxLoss

class SeqLabel(nn.Module):
    def __init__(self, data):
        super(SeqLabel, self).__init__()
        self.use_crf = data.use_crf
        print("build sequence labeling network...")
        print("use_char: ", data.use_char)
        if data.use_char:
            print("char feature extractor: ", data.char_feature_extractor)
        print("word feature extractor: ", data.word_feature_extractor)
        print("use crf: ", self.use_crf)

        self.gpu = data.HP_gpu and torch.cuda.is_available()
        self.average_batch = data.average_batch_loss
        ## add two more label for downlayer lstm, use original label size for CRF
        self.bilstm_flag = data.HP_bilstm
        if self.bilstm_flag:
            self.lstm_hidden = data.HP_hidden_dim // 2
        label_size = data.label_alphabet_size
        if data.use_crf:
            data.label_alphabet_size += 2
        self.word_hidden = WordSequence(data)

        if self.use_crf:
            self.crf = CRF(label_size, self.gpu)

        self.LM_use_sample_softmax = data.use_sample_softmax
        if self.LM_use_sample_softmax:
            self._LM_softmax = SampledSoftmaxLoss(num_words=data.word_alphabet_size, embedding_dim=self.lstm_hidden,
                                                num_samples=data.LM_sample_num, sparse=False, gpu=self.gpu)
        else:
            self._LM_softmax = LM_softmax(self.lstm_hidden, data.word_alphabet_size)

    def raw_loss(self, original_words_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, batch_entity, lm_seq_tensor, mask):
        batch_size, seq_len = word_inputs.size(0), word_inputs.size(1)
        outs_forward, outs_backward, cell_out, atten_probs = self.word_hidden('LM', original_words_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask)
        lm_forward_seq_tensor, lm_backward_seq_tensor = lm_seq_tensor
        total_loss = 0.0
        ##  compute entity loss
        loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
        cell_out = cell_out.view(batch_size * seq_len, -1)
        cell_score = F.log_softmax(cell_out, 1)
        cell_score_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(cell_score).to(cell_score.device)
        cell_score = torch.cat([cell_score_pad, cell_score], dim=-1)
        entity_loss = loss_function(cell_score, batch_entity.view(batch_size * seq_len))
        total_loss += entity_loss * 0.1
        ## compute probs loss
        atten_scores = atten_probs.view(batch_size * seq_len, -1)
        atten_probs = F.log_softmax(atten_scores, 1)
        atten_probs_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(atten_probs).to(atten_probs.device)
        atten_probs = torch.cat([atten_probs_pad, atten_probs], -1)
        atten_probs_loss = loss_function(atten_probs, batch_entity.view(batch_size * seq_len))
        total_loss += atten_probs_loss * 0.5
        ## compute LM loss
        if self.LM_use_sample_softmax:
            losses = []
            for idx, embedding, targets in ((0, outs_forward, lm_forward_seq_tensor),
                                            (1, outs_backward, lm_backward_seq_tensor)):
                non_masked_targets = targets.masked_select(mask) - 1
                # print(non_masked_targets)
                non_masked_embedding = embedding.masked_select(mask[:,:,None]).view(-1, self.lstm_hidden)
                # print(non_masked_targets)
                losses.append(self._LM_softmax(non_masked_embedding, non_masked_targets))
            total_loss += 0.5 * (losses[0] + losses[1])
            tag_seq_forward, tag_seq_backward = None, None
        else:
            loss_forward, score_forward = self._LM_softmax(outs_forward, lm_forward_seq_tensor, batch_size, seq_len)
            loss_backward, score_backward = self._LM_softmax(outs_backward, lm_backward_seq_tensor, batch_size, seq_len)
            total_loss += 0.5 * (loss_forward + loss_backward)
            _, tag_seq_forward = torch.max(score_forward, 1)
            _, tag_seq_backward = torch.max(score_backward, 1)
            tag_seq_forward = tag_seq_forward.view(batch_size, seq_len)
            tag_seq_backward = tag_seq_backward.view(batch_size, seq_len)
        ##compute perplexity
        length_mask = torch.sum(mask.float(), dim=1).float()
        # num = length_mask.sum(0).data.numpy()[0]
        num = length_mask.sum(0).item()
        if num != 0:
            perplexity = total_loss / num
        else:
            perplexity = 0.0
        if self.average_batch:
            if batch_size != 0:
                total_loss = total_loss / batch_size
            else:
                total_loss = 0

        return total_loss, perplexity, tag_seq_forward, tag_seq_backward

    def calculate_loss(self, original_words_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, batch_entity, mask):
        outs, cell_out, atten_scores = self.word_hidden('NER', original_words_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            # self.crf._calculate_con_P(outs, mask)
            total_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
            _, tag_seq = torch.max(score, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
        ## entity_cell loss
        loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
        cell_out = cell_out.view(batch_size * seq_len, -1)
        cell_score = F.log_softmax(cell_out, 1)
        cell_score_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(cell_score).to(cell_score.device)
        cell_score = torch.cat([cell_score_pad, cell_score], dim=-1)
        entity_loss = loss_function(cell_score, batch_entity.view(batch_size * seq_len))
        _, entity_seq = torch.max(cell_score, 1)
        entity_seq = entity_seq.view(batch_size, seq_len)
        total_loss += entity_loss * 0.1
        ## atten_probs loss
        atten_scores = atten_scores.view(batch_size * seq_len, -1)
        atten_probs = F.log_softmax(atten_scores, 1)
        atten_probs_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(atten_probs).to(atten_probs.device)
        atten_probs = torch.cat([atten_probs_pad, atten_probs], -1)
        atten_probs_loss = loss_function(atten_probs, batch_entity.view(batch_size * seq_len))
        _, atten_probs_seq = torch.max(atten_probs, 1)
        atten_probs_seq = atten_probs_seq.view(batch_size, seq_len)
        total_loss += atten_probs_loss * 0.5

        if self.average_batch:
            total_loss = total_loss / batch_size
        return total_loss, tag_seq, entity_seq, atten_probs_seq


    def forward(self, original_words_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs, cell_out, atten_scores = self.word_hidden('NER', original_words_batch, word_inputs,feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq
        ## entity cell predict
        cell_out = cell_out.view(batch_size * seq_len, -1)
        cell_out_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(cell_out).to(cell_out.device)
        cell_out = torch.cat([cell_out_pad, cell_out], dim=-1)
        _, entity_seq = torch.max(cell_out, 1)
        entity_seq = entity_seq.view(batch_size, seq_len)
        entity_seq = mask.long() * entity_seq
        ## atten prob predict
        atten_scores = atten_scores.view(batch_size * seq_len, -1)
        atten_scores_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(atten_scores).to(atten_scores.device)
        atten_scores = torch.cat([atten_scores_pad, atten_scores], dim=-1)
        _, atten_scores_seq = torch.max(atten_scores, 1)
        atten_scores_seq = atten_scores_seq.view(batch_size, seq_len)
        atten_scores_seq = mask.long() * atten_scores_seq

        return tag_seq, entity_seq, atten_scores_seq



    def decode_nbest(self, original_word_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest):
        if not self.use_crf:
            print("Nbest output is currently supported only for CRF! Exit...")
            exit(0)
        outs, cell_out, atten_scores = self.word_hidden('NER', original_word_batch, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        scores, tag_seq = self.crf._viterbi_decode_nbest(outs, mask, nbest)
        ## entity cell predict
        cell_out = cell_out.view(batch_size * seq_len, -1)
        cell_out_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(cell_out).to(cell_out.device)
        cell_out = torch.cat([cell_out_pad, cell_out], dim=-1)
        _, entity_seq = torch.max(cell_out, 1)
        entity_seq = entity_seq.view(batch_size, seq_len)
        entity_seq = mask.long() * entity_seq
        ## atten prob predict
        atten_scores = atten_scores.view(batch_size * seq_len, -1)
        atten_scores_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(atten_scores).to(atten_scores.device)
        atten_scores = torch.cat([atten_scores_pad, atten_scores], dim=-1)
        _, atten_scores_seq = torch.max(atten_scores, 1)
        atten_scores_seq = atten_scores_seq.view(batch_size, seq_len)
        atten_scores_seq = mask.long() * atten_scores_seq
        return scores, tag_seq, entity_seq, atten_scores_seq

class LM_softmax(nn.Module):
    def __init__(self, hidden_size, target_size):
        super(LM_softmax, self).__init__()
        self.hidden_to_tag = nn.Linear(hidden_size, target_size)

    def forward(self, embedding, lm_seq_tensor):
        batch_size, seq_len = embedding.size(0), embedding.size(1)
        outputs = self.hidden_to_tag(embedding)
        loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
        outputs = outputs.view(batch_size * seq_len, -1)
        scores = F.log_softmax(outputs, 1)
        loss = loss_function(scores, lm_seq_tensor.view(batch_size * seq_len))

        return loss, scores