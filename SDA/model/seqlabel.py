# -*- coding: utf-8 -*-


from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wordsequence import WordSequence
from .crf import CRF

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

        label_size_S = data.label_alphabet_size_S
        label_size_T = data.label_alphabet_size_T
        ## add two more label for downlayer lstm, use original label size for CRF
        if data.use_crf:
            data.label_alphabet_size += 2
            data.label_alphabet_size_S += 2
            data.label_alphabet_size_T += 2

        self.word_hidden = WordSequence(data)

        if self.use_crf:
            self.crf_S = CRF(label_size_S, self.gpu)
            self.crf_T = CRF(label_size_T, self.gpu)


    def calculate_loss(self, original_words_batch, domain_tag, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, batch_entity, mask):
        outs, cell_out_all, atten_scores_all = self.word_hidden(original_words_batch, domain_tag, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            crf = self.crf_S if domain_tag == "Source" else self.crf_T
            # crf = self.crf_S
            total_loss = crf.neg_log_likelihood_loss(outs, mask, batch_label)
        else:
            loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
            outs = outs.view(batch_size * seq_len, -1)
            score = F.log_softmax(outs, 1)
            total_loss = loss_function(score, batch_label.view(batch_size * seq_len))
        ## entity prediction loss (unified directions)
        if cell_out_all is not None:
            loss_function = nn.NLLLoss(ignore_index=0, reduction='sum')
            cell_out = cell_out_all.contiguous().view(batch_size * seq_len, -1)
            cell_score = F.log_softmax(cell_out, 1)
            cell_score_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(cell_score).to(cell_score.device)
            cell_score = torch.cat([cell_score_pad, cell_score], dim=-1)
            entity_loss_all = loss_function(cell_score, batch_entity.view(batch_size * seq_len))
        else:
            entity_loss_all = 0.0

        ## atten probs loss (unified directions)
        if atten_scores_all is not None:
            atten_scores = atten_scores_all.contiguous().view(batch_size * seq_len, -1)
            atten_probs = F.log_softmax(atten_scores, 1)
            atten_probs_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(atten_probs).to(atten_probs.device)
            atten_probs = torch.cat([atten_probs_pad, atten_probs], -1)
            atten_probs_loss_all = loss_function(atten_probs, batch_entity.view(batch_size * seq_len))
        else:
            atten_probs_loss_all = 0.0

        if self.average_batch:
            total_loss = total_loss / batch_size
            entity_loss_all = entity_loss_all / batch_size
            atten_probs_loss_all = atten_probs_loss_all / batch_size

        return total_loss, entity_loss_all, atten_probs_loss_all


    def forward(self, original_words_batch, domain_tag, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
        outs, cell_out_all, atten_scores_all = self.word_hidden(original_words_batch, domain_tag, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            crf = self.crf_S if domain_tag == "Source" else self.crf_T
            # crf = self.crf_S
            scores, tag_seq = crf._viterbi_decode(outs, mask)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = mask.long() * tag_seq

        ## entity prediction (unified directions)
        if cell_out_all is not None:
            cell_out = cell_out_all.contiguous().view(batch_size * seq_len, -1)
            cell_out_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(cell_out).to(cell_out.device)
            cell_out = torch.cat([cell_out_pad, cell_out], dim=-1)
            _, entity_seq = torch.max(cell_out, 1)
            entity_seq = entity_seq.view(batch_size, seq_len)
            entity_seq_all = mask.long() * entity_seq
        else:
            entity_seq_all = None

        ## atten probs (unified directions)
        if atten_scores_all is not None:
            atten_scores = atten_scores_all.contiguous().view(batch_size * seq_len, -1)
            atten_scores_pad = torch.Tensor(batch_size * seq_len, 1).fill_(float('-inf')).type_as(atten_scores).to(atten_scores.device)
            atten_scores = torch.cat([atten_scores_pad, atten_scores], dim=-1)
            _, atten_scores_seq = torch.max(atten_scores, 1)
            atten_scores_seq = atten_scores_seq.view(batch_size, seq_len)
            atten_scores_seq_all = mask.long() * atten_scores_seq
        else:
            atten_scores_seq_all = None

        return tag_seq, entity_seq_all, atten_scores_seq_all



    def decode_nbest(self, original_words_batch, domain_tag, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, nbest, batch_entity):
        if not self.use_crf and nbest > 1:
            print("Nbest(N>1) output is currently supported only for CRF! Exit...")
            exit(0)
        outs, cell_out, atten_scores = self.word_hidden(original_words_batch, domain_tag, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, batch_entity)
        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        if self.use_crf:
            crf = self.crf_T if domain_tag == "Target" else self.crf_S
            scores, tag_seq = crf._viterbi_decode_nbest(outs, mask, nbest)
        else:
            outs = outs.view(batch_size * seq_len, -1)
            scores = F.softmax(outs, 1)[:,:,None]
            _, tag_seq = torch.max(outs, 1)
            tag_seq = tag_seq.view(batch_size, seq_len)
            ## filter padded position with zero
            tag_seq = (mask.long() * tag_seq)[:,:,None]
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

