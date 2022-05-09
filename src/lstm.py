import torch.nn as nn
import torch.nn.functional as F
import torch

# class Fullyconnected(nn.Sequential):


class LSTMCell(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        state_size = output_size

        in_out_size = input_size + output_size

        self.layer_forget_gate = nn.Sequential(
            nn.Linear(in_out_size, state_size),
            nn.Sigmoid()
        )

        self.layer_input_gate = nn.Sequential(
            nn.Linear(in_out_size, state_size),
            nn.Sigmoid()
        )

        self.layer_candidates = nn.Sequential(
            nn.Linear(in_out_size, state_size),
            nn.Tanh()
        )

        self.layer_output_gate = nn.Sequential(
            nn.Linear(in_out_size, output_size),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor, out_tm1: torch.Tensor, state_tm1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [TODO:description]

        :param x_t torch.Tensor[n_batch, n_in]: the input for step t
        :param out_tm1 torch.Tensor[n_batch, n_out]: the output for the previous step (also called $h_{t-1}$ )
        :param state_tm1[n_batch, n_state] torch.Tensor: the state after the previous step (also called $C_{t-1}$ )
        :rtype tuple[torch.Tensor, torch.Tensor]: the output and the state after this step

        """
        xh = torch.cat((x, out_tm1), dim=-1)
        new_state = state_tm1 * self.layer_forget_gate(xh) \
            + self.layer_input_gate(xh) * self.layer_candidates(xh)
        
        output_t = self.layer_output_gate(xh) * F.tanh(new_state)

        return output_t, new_state


class LSTM(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size


        self.cell = LSTMCell(input_size, output_size)


    def forward(self, tokens: torch.Tensor):
        """
        :param tokens torch.Tensor: list of tokens of size (n_tokens, batch_size, input_size)

        returns (n_token, batch_size, output_size)
        """
        state = torch.zeros(())
        out = torch.zeros(())

        outputs = []

        for token in tokens:
            out, state = self.cell(token, out, state)
            outputs.append(out)

        return torch.stack(outputs, dim=0)


