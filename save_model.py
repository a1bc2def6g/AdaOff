import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)

model = CustomLinear(10, 5)
torch.save(model, 'custom_linear_model.pth')