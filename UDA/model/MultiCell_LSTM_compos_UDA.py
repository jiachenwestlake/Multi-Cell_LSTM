import copy
import math
import torch
from torch import nn
from torch.nn import functional, init
import torch.nn.functional as F
import numpy as np

class Compute_Gate(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, use_bias=True, device='cpu'):
        super(Compute_Gate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.use_bias = use_bias
        self.device = device

        cell_x = nn.Linear(self.input_size, 2 * self.hidden_size, bias=use_bias)
        cell_h = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        self.cell_x_list = nn.ModuleList([copy.deepcopy(cell_x) for _ in range(self.cell_num)])
        self.cell_h_list = nn.ModuleList([copy.deepcopy(cell_h) for _ in range(self.cell_num)])

    def forward(self, input_word, hidden_states):
        batch_size = input_word.size(0)
        net_cell_list = []
        for cell_idx in range(self.cell_num):
            net_input_x = self.cell_x_list[cell_idx](input_word)  # (batch_size, 3 * hidden_size)
            net_input_h = self.cell_h_list[cell_idx](hidden_states)
            net_cell = net_input_x + net_input_h
            net_cell_list.append(net_cell.unsqueeze(0))

        net_cells = torch.cat(net_cell_list, dim=0).contiguous().transpose(0, 1)
        input_gate, cell_states = torch.split(net_cells, split_size_or_sections=self.hidden_size, dim=-1)
        cell_input = torch.tanh(cell_states)
        input_gate = torch.sigmoid(input_gate)

        return cell_input, input_gate


class Compute_Add_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, device='cpu'):
        super(Compute_Add_Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.device = device

        self.input_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)
        self.cell_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)

        self.atten_v = nn.Parameter(
            torch.Tensor(2 * self.hidden_size)
        )

        self.reset_parameter()

    def reset_parameter(self):
        bound = 1.0 / math.sqrt(self.atten_v.size(0))
        init.uniform_(self.atten_v, -bound, bound)

    def forward(self, cell_state_c, cell_states):
        batch_size = cell_state_c.size(0)
        atten_input = cell_state_c.unsqueeze(1).expand(batch_size, self.cell_num, self.hidden_size)

        net_atten_input = self.input_proj(atten_input)
        net_atten_cell = self.cell_proj(cell_states)

        scores = torch.tanh(net_atten_input + net_atten_cell)
        scores = torch.einsum('h,bch->bc', (self.atten_v, scores)) # (batch_size, cell_num)
        # scores = scores.float().masked_fill(1 - entity_mask[None,:,None], float('-inf')).type_as(scores)
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

    def forward(self, input, hidden_states):
        net_input_x = self.input_x(input)
        net_input_h = self.input_h(hidden_states)

        net_input = net_input_x + net_input_h

        input_gate, output_gate, cell_states = torch.split(net_input, split_size_or_sections=self.hidden_size, dim=-1)

        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)
        cell_states = torch.tanh(cell_states)

        return input_gate, output_gate, cell_states


class MultiCell_layer(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, use_bias=True, device='cpu'):
        super(MultiCell_layer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.use_bias = use_bias
        self.device = device

        self.compute_gate = Compute_Gate(input_size, hidden_size, cell_num, use_bias=use_bias, device=device)
        self.composition_cell = Composition_Cell(input_size, hidden_size, use_bias=use_bias, device=device)
        self.compute_atten = Compute_Add_Attention(input_size, hidden_size, cell_num, device=device)


    def forward(self, input_word, hidden_states, cell_state_pre):
        """
        :param input_word: (batch_size, input_size)
        :param hidden_states: (batch_size, hidden_size)
        :param cell_state_pre: (batch_size, hidden_size)
        :return:
        """
        batch_size, hidden_size = hidden_states.size(0), hidden_states.size(1)
        ## compute entity cells (batch_size, cell_num, hidden_size)
        cell_states, input_gates = self.compute_gate(input_word, hidden_states)
        cell_states = input_gates * cell_states + (1 - input_gates) * cell_state_pre.unsqueeze(1).expand_as(cell_states)
        ## compute the center cell
        input_gate, output_gate, cell_state_c = self.composition_cell(input_word, hidden_states)
        ## center cell attention entity cells
        atten_cell, probs = self.compute_atten(cell_state_c, cell_states)

        cell_state_c = (1 - input_gate) * cell_state_pre + input_gate * atten_cell

        hidden_outputs = output_gate * torch.tanh(cell_state_c)

        return hidden_outputs, cell_state_c, cell_states, probs


class MultiCellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cell_num, left2right=True, use_bias=True, gpu=True):
        super(MultiCellLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_num = cell_num
        self.left2right = left2right
        self.use_bias = use_bias

        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        self.layer = MultiCell_layer(self.input_size, self.hidden_size, self.cell_num, self.use_bias, self.device)

    def forward(self, input, mask, hidden_states=None):
        # entity_mask = torch.Tensor(entity_mask).byte().to(self.device) # (cell_num) [1:'O', 1/0:'PER', ...]
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
            mask_t = mask.transpose(0, 1)
            mask_t_r = mask_t[seq_list]
            mask_r = mask_t_r.transpose(0, 1)
            ## reorder the input
            input_value = input.masked_select(mask[:,:,None])
            input_zero = torch.zeros_like(input, dtype=input.dtype).to(self.device)
            input = input_zero.masked_scatter(mask_r[:,:,None], input_value)

        input = input.transpose(0, 1) # (seq_len, batch_size, input_size)
        hidden_list = []
        cell_states_list = []
        atten_probs_list = []
        for pos_idx in seq_list:
            hidden_states, cell_state_c, cell_states, atten_probs = self.layer(input[pos_idx], hidden_states, cell_state_c)
            hidden_list.append(hidden_states.unsqueeze(0))
            cell_states_list.append(cell_states.unsqueeze(0))
            atten_probs_list.append(atten_probs.unsqueeze(0))
            # print(hidden_outputs.size())

        if not self.left2right:
            hidden_list = list(reversed(hidden_list))
            cell_states_list = list(reversed(cell_states_list))
            atten_probs_list = list(reversed(atten_probs_list))
        # print(hidden_list)
        hidden_output_seq = torch.cat(hidden_list, dim=0).transpose(0, 1)
        cell_states_seq = torch.cat(cell_states_list, dim=0).transpose(0, 1)
        atten_probs_seq = torch.cat(atten_probs_list, dim=0).transpose(0, 1)
        if not self.left2right: ## reorder the output
            # print(hidden_output_seq.size(), mask_r.size())
            hidden_output_seq_value = hidden_output_seq.masked_select(mask_r[:,:,None])
            output_zero = torch.zeros_like(hidden_output_seq, dtype=hidden_output_seq.dtype).to(self.device)
            hidden_output_seq = output_zero.masked_scatter(mask[:,:,None], hidden_output_seq_value)
            ##cell states
            cell_states_seq_value = cell_states_seq.masked_select(mask_r[:,:,None,None])
            cell_zero = torch.zeros_like(cell_states_seq, dtype=hidden_output_seq.dtype).to(self.device)
            cell_states_seq = cell_zero.masked_scatter(mask[:,:,None,None], cell_states_seq_value)
            ## atten probs
            atten_probs_value = atten_probs_seq.masked_select(mask_r[:,:,None])
            atten_probs_zero = torch.zeros_like(atten_probs_seq, dtype=hidden_output_seq.dtype).to(self.device)
            atten_probs_seq = atten_probs_zero.masked_scatter(mask[:,:,None], atten_probs_value)

        return hidden_output_seq, cell_states_seq, atten_probs_seq

