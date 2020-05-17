import copy
import math
import torch
from torch import nn
from torch.nn import functional, init

class Cell_to_Tag(nn.Module):
    """ Transfer the cell output of MultiCellLSTM into softmax input """
    def __init__(self, task, hidden_size, label_num, entity_num, pos_num, use_bias=True):
        super(Cell_to_Tag, self).__init__()
        self.task = task
        self.hidden_size = hidden_size
        self.pos_num = pos_num
        self.entity_num = entity_num
        self.label_num = label_num # number of labels + 1
        self.use_bias = use_bias
        if self.task == 'POSTag':
            self.weight_CT = nn.Parameter( # requires_grad = True
                torch.Tensor(self.label_num, self.hidden_size)
            )
            if self.use_bias:
                self.bias_CT = nn.Parameter(
                    torch.Tensor(self.label_num)
                )
        elif self.task == 'NER':
            # self.weight_CT = nn.Parameter( # requires_grad = True
            #     torch.Tensor(self.entity_num, self.hidden_size, self.pos_num)
            # )
            # if self.use_bias:
            #     self.bias_CT = nn.Parameter(
            #         torch.Tensor(self.entity_num * self.pos_num)
            #     )
            layer = nn.Linear(self.hidden_size, self.pos_num)
            self.hidden2label_list = nn.ModuleList([copy.deepcopy(layer) for _ in range(self.entity_num)])

    #     self.reset_parameter()
    #
    # def reset_parameter(self):
    #     # stdv = 1. / math.sqrt(self.weight_CT.size(1))
    #     # self.weight_CT.data.uniform_(-stdv, stdv)
    #     init.kaiming_uniform_(self.weight_CT, a=math.sqrt(5))
    #     if self.use_bias:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_CT)
    #         bound = 1.0 / math.sqrt(fan_in)
    #         init.uniform_(self.bias_CT, -bound, bound)

    def forward(self, hidden_outputs):
        """
        :param hidden_outputs: (batch_size, seq_len, cell_num, hidden_size)
        :return: (batch_size, seq_len, label_size)
        """
        batch_size = hidden_outputs.size(0)
        seq_len = hidden_outputs.size(1)
        if self.task == 'POSTag':
            bias_CT_batch = self.bias_CT[None,None,:].expand(batch_size, seq_len, self.label_num)
            label_outputs = torch.einsum('bsch,ch->bsc', (hidden_outputs, self.weight_CT))
            if self.use_bias:
                label_outputs += bias_CT_batch
        elif self.task == 'NER':
            # bias_CT_batch = self.bias_CT[None,None,:].expand(batch_size, seq_len, self.entity_num * self.pos_num)
            # label_outputs = torch.einsum('bsch,chp->bscp', (hidden_outputs, self.weight_CT))
            # label_outputs = label_outputs.contiguous().view(batch_size, seq_len, self.entity_num * self.pos_num)
            # if self.use_bias:
            #     label_outputs += bias_CT_batch
            # label_outputs = label_outputs[:, :, 2:]
            label_outputs_list = []
            for entity_idx in range(self.entity_num):
                label_outputs_list.append(self.hidden2label_list[entity_idx](hidden_outputs[:,:,entity_idx,:])) #(b
            label_outputs = torch.cat(label_outputs_list, dim=-1) # (batch_size, seq_len, pos_num * entit
            label_outputs = label_outputs[:,:,2:] # (batch_size, seq_len, pos_num*(entity_num-1) + 2)
        assert(label_outputs.size(-1) == self.label_num)

        return label_outputs


