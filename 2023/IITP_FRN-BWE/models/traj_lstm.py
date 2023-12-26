import torch
from torch import nn
import numpy as np

class TrajLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_layers=1, dropout=0, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        time_lstm = [nn.LSTM(input_size, hidden_size)]
        for _ in range(num_layers - 1):
            time_lstm.append(nn.LSTM(hidden_size, hidden_size))
        self.time_lstm = nn.Sequential(*time_lstm)
        self.drop = nn.Dropout(dropout)
        self.depth_lstm = nn.LSTM(hidden_size, hidden_size)
        self.layernorms = nn.ModuleList(
            [torch.nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, input, hidden=None):
        if self.batch_first:
            # print(input.shape)
            input = input.transpose(0, 1)
            # input = input.transpose(0, 2, 1)
            # print(input.shape)
            # exit()
        time_output = input
        time_results = []
        if hidden is None:
            hidden = [None for _ in self.time_lstm]
        else:
            all_h, all_c = hidden
            hidden = [(h.unsqueeze(0), c.unsqueeze(0))
                      for h, c in zip(all_h, all_c)]
        next_hidden = []
        next_cell = []
        for lstm, state, layernorm in zip(self.time_lstm, hidden, self.layernorms):
            
            # print(layernorm)
            # print(time_output.shape)
            # exit()
            # seq X bs X hidden
            time_output = layernorm(time_output)
            time_output, (next_h, next_c) = lstm(time_output, state)
            next_hidden.append(next_h)
            next_cell.append(next_c)
            time_output = self.drop(time_output)
            time_results.append(time_output)

        time_results = torch.stack(time_results)  # depth X seq X bs X hidden
        depth, seq, bs, hidden = time_results.size()
        _, (depth_h, depth_c) = self.depth_lstm(
            time_results.view(depth, seq*bs, hidden)
        )
        output = depth_c  # seq*bs X hidden
        output = output.view(seq, bs, hidden) + time_output
        next_state = (torch.stack(next_hidden[::-1]).squeeze(1),
                      torch.stack(next_cell[::-1]).squeeze(1))
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, next_state

    def flatten_parameters(self):
        for lstm in self.time_lstm:
            lstm.flatten_parameters()
        self.depth_lstm.flatten_parameters()


class LT_LSTM(TrajLSTM):
    def forward(self, input, hidden=None):
        time_output = input
        hidden = [None for _ in self.time_lstm]
        depth_next = None
        for lstm, cur_hidden in zip(self.time_lstm, hidden):
            time_output, _ = lstm(time_output, None)  # seq X bs X hidden
            time_output = self.drop(time_output)
            depth_out, depth_next = self.depth_lstm(
                time_output.view(1, -1, self.hidden_size),
                depth_next,
            )
            # use the output of time_lstm as the input of the next layer
            # of time lstm
            time_output = depth_out.view_as(time_output)

        return time_output, _
        return time_output, None


def test_traj_lstm():
    traj_lstm = TrajLSTM(256, 256, 6).cuda()
    fake_input = torch.zeros(30, 30, 256).cuda()
    # x = np.ndarray([traj_lstm(fake_input)])
    # print(traj_lstm(fake_input))
    # exit()
    # print(x.mean().item())
    # print(traj_lstm(fake_input).mean().item())
    # print(traj_lstm(fake_input).mean().item())


if __name__ == '__main__':
    test_traj_lstm()