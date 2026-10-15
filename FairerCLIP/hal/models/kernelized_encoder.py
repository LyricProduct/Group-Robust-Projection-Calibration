import torch
from torch import nn
from torch.nn.parameter import Parameter

__all__ = ["KernelizedEncoder"]


class KernelizedEncoder(nn.Module):
    def __init__(self, U, w, b):
        super().__init__()
        self.dtype = U.dtype

        if isinstance(U, list):
            self.U = Parameter(torch.zeros(*tuple(U)), requires_grad=False)
        else:
            self.U = Parameter(U)

        if isinstance(w, list):
            self.w = Parameter(torch.zeros(*tuple(w)), requires_grad=False)
        else:
            self.w = Parameter(w)

        if isinstance(b, list):
            self.b = Parameter(torch.zeros(*tuple(b)), requires_grad=False)
        else:
            self.b = Parameter(b)

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        phi_x = torch.sqrt(
            torch.tensor([2.0 / len(self.w)], device=x.device)
        ) * torch.cos(
            torch.mm(x, self.w.t().to(dtype=self.dtype)) + self.b.to(dtype=self.dtype)
        )
        z = torch.mm(phi_x, self.U)
        return z.float()
