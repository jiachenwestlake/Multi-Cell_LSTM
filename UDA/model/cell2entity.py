import copy
import math
import torch
from torch import nn

class Cell_to_Entity(nn.Module):
    def __init__(self, hidden_size, entity_num, use_bias=True):
        super(Cell_to_Entity, self).__init__()

        self.hidden_size = hidden_size
        self.entity_num = entity_num
        self.use_bias = use_bias

        layer = nn.Linear(self.hidden_size, 1, bias=self.use_bias)

        self.hidden2entity_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.entity_num)])

    def forward(self, cell_states):
        batch_size = cell_states.size(0)
        seq_len = cell_states.size(1)

        entity_output_list = []
        # pad = torch.Tensor(batch_size, seq_len, 1).fill_(float('-inf')).type_as(cell_states.dtype).to(cell_states.device)
        # entity_output_list.append(pad)
        for entity_idx in range(self.entity_num):
            entity_output_list.append(self.hidden2entity_list[entity_idx](cell_states[:,:,entity_idx,:]))

        entity_output = torch.cat(entity_output_list, dim=-1)

        return entity_output
