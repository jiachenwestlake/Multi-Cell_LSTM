import copy
import math
import torch
from torch import nn

class Cell_to_Entity(nn.Module):
    def __init__(self, hidden_size, entity_num, entity_mask_S, entity_mask_T, use_bias=True):
        super(Cell_to_Entity, self).__init__()

        self.hidden_size = hidden_size
        self.entity_num = entity_num
        self.entity_mask_S = entity_mask_S
        self.entity_mask_T = entity_mask_T
        self.use_bias = use_bias

        layer = nn.Linear(self.hidden_size, 1, bias=self.use_bias)

        self.hidden2entity_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.entity_num)])
        # self.hidden2entity_list_S = nn.ModuleList([copy.deepcopy(layer) for _ in range(sum(self.entity_mask_S))])
        # self.hidden2entity_list_T = nn.ModuleList([copy.deepcopy(layer) for _ in range(sum(self.entity_mask_T))])
        self.hidden2entity_list_S = nn.ModuleList([self.hidden2entity_list[idx] for idx in range(self.entity_num) if self.entity_mask_S[idx] == 1])
        self.hidden2entity_list_T = nn.ModuleList([self.hidden2entity_list[idx] for idx in range(self.entity_num) if self.entity_mask_T[idx] == 1])

    def forward(self, domain_tag, cell_states):
        batch_size = cell_states.size(0)
        seq_len = cell_states.size(1)
        entity_num = sum(self.entity_mask_S) if domain_tag == "Source" else sum(self.entity_mask_T)
        hidden2entity_list = self.hidden2entity_list_S if domain_tag == "Source" else self.hidden2entity_list_T
        entity_output_list = []
        # pad = torch.zeros(batch_size, seq_len, 1, dtype=cell_states.dtype).to(cell_states.device)
        # entity_output_list.append(pad)
        for entity_idx in range(entity_num):
            entity_output_list.append(hidden2entity_list[entity_idx](cell_states[:,:,entity_idx,:]))

        entity_output = torch.cat(entity_output_list, dim=-1)

        return entity_output
