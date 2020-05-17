import torch
from torch import nn
from torch.autograd.variable import Variable
from torch.nn import init

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_ih = nn.Linear(in_features=input_size,
                                   out_features=4 * hidden_size)
        self.linear_hh = nn.Linear(in_features=hidden_size,
                                   out_features=4 * hidden_size,
                                   bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.linear_ih.weight.data)
        init.constant_(self.linear_ih.bias.data, val=0)
        init.orthogonal_(self.linear_hh.weight.data)

    def forward(self, x, state):
        if state is None:
            batch_size = x.size(0)
            zero_state = Variable(x.data.new(batch_size, self.hidden_size).zero_())
            state = (zero_state, zero_state)
        h, c = state
        lstm_vector = self.linear_ih(x) + self.linear_hh(h)
        i, f, g, o = lstm_vector.chunk(chunks=4, dim=1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * self.dropout(torch.tanh(g))
        new_h = torch.tanh(new_c) * torch.sigmoid(o)
        new_state = (new_h, new_c)

        return new_h, new_state

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, bidirectional=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_cell = LSTMCell(self.input_size, self.hidden_size, self.dropout)
        if self.bidirectional:
            self.rnn_cell_back = LSTMCell(self.input_size, self.hidden_size, self.dropout)

    def forward(self, inputs_emb, mask, hidden):
        batch_size, max_seq_len = inputs_emb.size(0), inputs_emb.size(1)
        inputs = inputs_emb.transpose(0, 1)  # (seq_length, batch_size, input_size)
        rnn_outputs = []
        state = hidden
        for t in range(max_seq_len):
            output, state = self.rnn_cell.forward(x=inputs[t], state=state)
            rnn_outputs.append(output)
        rnn_outputs = torch.stack(rnn_outputs, dim=0).transpose(0, 1)  # (batch_size, seq_length, hidden_size)
        if self.bidirectional:
            seq_list = list(reversed(list(range(max_seq_len))))
            batch_sent_len = mask.long().sum(1)
            index_b, index_f = [], []
            for sent_idx in range(batch_size):
                sent_len = batch_sent_len[sent_idx].item()
                index_b.append(list(range(sent_len, max_seq_len)) + list(range(sent_len)))
                index_f.append(list(range(max_seq_len - sent_len, max_seq_len)) + list(range(max_seq_len - sent_len)))
            index_b = torch.LongTensor(index_b).to(inputs_emb.device)
            index_f = torch.LongTensor(index_f).to(inputs_emb.device)
            inputs_emb = torch.gather(inputs_emb, dim=1, index=index_b[:, :, None].expand_as(inputs_emb))
            inputs_back = inputs_emb.transpose(0, 1)  # (seq_len, batch_size, input_size)
            rnn_outputs_back = []
            state = hidden
            for t in seq_list:
                output, state = self.rnn_cell_back.forward(x=inputs_back[t], state=state)
                rnn_outputs_back.append(output)
            rnn_outputs_back = list(reversed(rnn_outputs_back))
            rnn_outputs_back = torch.stack(rnn_outputs_back, dim=0).transpose(0, 1)  # (batch_size, seq_length, hidden_size)
            rnn_outputs_back = torch.gather(rnn_outputs_back, dim=1, index=index_f[:, :, None].expand_as(rnn_outputs_back))
            rnn_outputs = torch.cat([rnn_outputs, rnn_outputs_back], dim=-1)

        return rnn_outputs