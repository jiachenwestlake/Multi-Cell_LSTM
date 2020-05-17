import copy
import math
import torch
from torch import nn
from torch.nn import functional, init
import torch.nn.functional as F
import numpy as np

class Compute_Gate(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, entity_mask_S, entity_mask_T, use_bias=True, device='cpu'):
        super(Compute_Gate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.entity_mask_S = entity_mask_S
        self.entity_mask_T = entity_mask_T
        self.entity_num_S = sum(entity_mask_S)
        self.entity_num_T = sum(entity_mask_T)
        self.use_bias = use_bias
        self.device = device

        cell_x = nn.Linear(self.input_size, 2 * self.hidden_size, bias=use_bias)
        cell_h = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        self.cell_list_x = nn.ModuleList([copy.deepcopy(cell_x) for _ in range(self.cell_num)])
        self.cell_list_h = nn.ModuleList([copy.deepcopy(cell_h) for _ in range(self.cell_num)])

        self.cell_list_S_x = nn.ModuleList([self.cell_list_x[idx] for idx in range(self.cell_num) if self.entity_mask_S[idx] == 1])
        self.cell_list_S_h = nn.ModuleList([self.cell_list_h[idx] for idx in range(self.cell_num) if self.entity_mask_S[idx] == 1])
        self.cell_list_T_x = nn.ModuleList([self.cell_list_x[idx] for idx in range(self.cell_num) if self.entity_mask_T[idx] == 1])
        self.cell_list_T_h = nn.ModuleList([self.cell_list_h[idx] for idx in range(self.cell_num) if self.entity_mask_T[idx] == 1])


    def forward(self, domain_tag, input_word, hidden_states):
        batch_size = input_word.size(0)
        net_cell_list = []
        cell_num = self.entity_num_S if domain_tag == "Source" else self.entity_num_T
        for cell_idx in range(cell_num):
            if domain_tag == "Source":
                net_input_x = self.cell_list_S_x[cell_idx](input_word)  # (batch_size, 3 * hidden_size)
                net_input_h = self.cell_list_S_h[cell_idx](hidden_states)
            elif domain_tag == "Target":
                net_input_x = self.cell_list_T_x[cell_idx](input_word)  # (batch_size, 3 * hidden_size)
                net_input_h = self.cell_list_T_h[cell_idx](hidden_states)
            net_cell = net_input_x + net_input_h
            net_cell_list.append(net_cell.unsqueeze(0))

        net_cells = torch.cat(net_cell_list, dim=0).contiguous().transpose(0, 1)
        input_gate, cell_states = torch.split(net_cells, split_size_or_sections=self.hidden_size, dim=-1)
        cell_input = torch.tanh(cell_states)
        input_gate = torch.sigmoid(input_gate)

        return cell_input, input_gate


class Compute_Add_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, entity_mask_S, entity_mask_T, device='cpu'):
        super(Compute_Add_Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.cell_num_S = sum(entity_mask_S)
        self.cell_num_T = sum(entity_mask_T)
        self.device = device

        self.input_proj_S = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)
        self.cell_proj_S = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        self.atten_v_S = nn.Parameter(
            torch.Tensor(2 * self.hidden_size)
        )

        self.input_proj_T = copy.deepcopy(self.input_proj_S)
        self.cell_proj_T = copy.deepcopy(self.cell_proj_S)
        self.atten_v_T = copy.deepcopy(self.atten_v_S)

        self.reset_parameter()

    def reset_parameter(self):
        bound = 1.0 / math.sqrt(self.atten_v_S.size(0))
        init.uniform_(self.atten_v_S, -bound, bound)
        bound = 1.0 / math.sqrt(self.atten_v_T.size(0))
        init.uniform_(self.atten_v_T, -bound, bound)

    def forward(self, domain_tag, cell_state_c, cell_states):
        batch_size = cell_state_c.size(0)
        cell_num = self.cell_num_S if domain_tag == "Source" else self.cell_num_T
        input_proj = self.input_proj_S if domain_tag == "Source" else self.input_proj_T
        cell_proj = self.input_proj_S if domain_tag == "Source" else self.cell_proj_T
        atten_v = self.atten_v_S if domain_tag == "Source" else self.atten_v_T

        atten_input = cell_state_c.unsqueeze(1).contiguous().expand(batch_size, cell_num, self.hidden_size)

        net_atten_input = input_proj(atten_input)
        net_atten_cell = cell_proj(cell_states)

        scores = torch.tanh(net_atten_input + net_atten_cell)
        scores = torch.einsum('h,bch->bc', (atten_v, scores)) # (batch_size, cell_num)

        probs = F.softmax(scores, dim=1) #(batch_size, cell_num)

        atten_output = torch.einsum('bc,bch->bh', (probs, cell_states))

        return atten_output, probs


class Composition_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True, device='cpu'):
        super(Composition_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.device = device

        self.input_x = nn.Linear(self.input_size, 3 * self.hidden_size, bias=use_bias)
        self.input_h = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)

    def forward(self, domain_tag, input, hidden_states):

        net_input_x = self.input_x(input)
        net_input_h = self.input_h(hidden_states)

        net_input = net_input_x + net_input_h

        input_gate, output_gate, cell_state_c = torch.split(net_input, split_size_or_sections=self.hidden_size, dim=-1)

        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_state_c = torch.tanh(cell_state_c)

        return input_gate, output_gate, cell_state_c



class MultiCell_layer(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, entity_mask_S, entity_mask_T, use_bias=True, device='cpu'):
        super(MultiCell_layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.entity_mask_S = entity_mask_S
        self.entity_mask_T = entity_mask_T
        self.use_bias = use_bias
        self.device = device

        self.compute_gate = Compute_Gate(input_size, hidden_size, cell_num, entity_mask_S, entity_mask_T, use_bias=use_bias, device=device)
        self.composition_cell = Composition_Cell(input_size, hidden_size, use_bias=use_bias, device=device)
        self.compute_atten = Compute_Add_Attention(input_size, hidden_size, cell_num, entity_mask_S, entity_mask_T, device=device)


    def forward(self, domain_tag, input_word, hidden_states, cell_state_pre):
        """
        :param input_word: (batch_size, input_size)
        :param hidden_states: (batch_size, hidden_size)
        :param cell_state_pre: (batch_size, hidden_size)
        :return:
        """
        batch_size, hidden_size = hidden_states.size(0), hidden_states.size(1)
        ## compute entity cells (batch_size, cell_num, hidden_size)
        cell_states, input_gates = self.compute_gate(domain_tag, input_word, hidden_states)
        cell_states = input_gates * cell_states + (1 - input_gates) * cell_state_pre.unsqueeze(1).expand_as(cell_states)
        ## compute the center cell
        input_gate, output_gate, cell_state_c = self.composition_cell(domain_tag, input_word, hidden_states)
        ## center cell attention entity cells
        atten_cell, probs = self.compute_atten(domain_tag, cell_state_c, cell_states)

        cell_state_c = (1 - input_gate) * cell_state_pre + input_gate * atten_cell

        hidden_outputs = output_gate * torch.tanh(cell_state_c)

        return hidden_outputs, cell_state_c, cell_states, probs


class MultiCellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, entity_mask_S, entity_mask_T, left2right=True, use_bias=True, gpu=True):
        super(MultiCellLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.cell_num_S = sum(entity_mask_S)
        self.cell_num_T = sum(entity_mask_T)
        self.entity_mask_S = entity_mask_S
        self.entity_mask_T = entity_mask_T
        self.left2right = left2right
        self.use_bias = use_bias

        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        self.layer = MultiCell_layer(self.input_size, self.hidden_size, self.cell_num, self.entity_mask_S, self.entity_mask_T, self.use_bias, self.device)

    def forward(self, domain_tag, input, mask, hidden_states=None):
        batch_size = input.size(0)
        seq_len = input.size(1)

        if hidden_states is not None:
            hidden_states = hidden_states
        else:
            cell_state_c = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype).to(self.device)
            hidden_states = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype).to(self.device)

        seq_list = list(range(seq_len))
        if not self.left2right:
            seq_list = list(reversed(seq_list))
            batch_seq_len = mask.long().sum(1) #(batch_size)
            index_b, index_f = [], []
            for sent_idx in range(batch_size):
                sent_len = batch_seq_len[sent_idx].item()
                index_b.append(list(range(sent_len, seq_len)) + list(range(sent_len)))
                index_f.append(list(range(seq_len-sent_len, seq_len)) + list(range(seq_len-sent_len)))
            index_b = torch.LongTensor(index_b).to(self.device)
            index_f = torch.LongTensor(index_f).to(self.device)
            input = torch.gather(input, dim=1, index=index_b[:,:,None].expand_as(input))


        input = input.transpose(0, 1) # (seq_len, batch_size, input_size)
        hidden_list = []
        cell_states_list = []
        atten_probs_list = []
        for pos_idx in seq_list:
            hidden_states, cell_state_c, cell_states, atten_probs = self.layer(domain_tag, input[pos_idx], hidden_states, cell_state_c)
            hidden_list.append(hidden_states.unsqueeze(0))
            cell_states_list.append(cell_states.unsqueeze(0))
            atten_probs_list.append(atten_probs.unsqueeze(0))

        if not self.left2right:
            hidden_list = list(reversed(hidden_list))
            cell_states_list = list(reversed(cell_states_list))
            atten_probs_list = list(reversed(atten_probs_list))
        # print(hidden_list)
        hidden_output_seq = torch.cat(hidden_list, dim=0).transpose(0, 1)
        cell_states_seq = torch.cat(cell_states_list, dim=0).transpose(0, 1)
        atten_probs_seq = torch.cat(atten_probs_list, dim=0).transpose(0, 1)
        if not self.left2right: ## reorder the output
            hidden_output_seq = torch.gather(hidden_output_seq, dim=1, index=index_f[:,:,None].expand_as(hidden_output_seq))
            ##cell states
            cell_states_seq = torch.gather(cell_states_seq, dim=1, index=index_f[:,:,None,None].expand_as(cell_states_seq))
            ## atten probs
            atten_probs_seq = torch.gather(atten_probs_seq, dim=1, index=index_f[:,:,None].expand_as(atten_probs_seq))

        return hidden_output_seq, cell_states_seq, atten_probs_seq

