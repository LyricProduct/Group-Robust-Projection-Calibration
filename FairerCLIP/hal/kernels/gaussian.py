import math
import torch

__all__ = ["RFFGaussian"]


class RFFGaussian:
    def __init__(self, sigma=None, rff_dim=200, sigma_numel_max=9000):
        self.sigma = sigma
        self.rff_dim = rff_dim
        self.numel_max = sigma_numel_max
        self.w = None
        self.b = None

    def _calc_w_b(self, x):
        dim_x = x.shape[1]
        if self.sigma is None:
            n = min(self.numel_max, x.shape[0])
            rand = torch.randperm(n)
            x_samp = x[rand, :]
            x_samp = x_samp[0:n, :]
            dist = torch.cdist(x_samp, x_samp, p=2) ** 2
            dist = torch.triu(dist, diagonal=1)
            if (dist > 0).sum():
                sigma = torch.sqrt(0.5 * torch.median(dist[dist > 0]))
            else:
                sigma = 1

        else:
            sigma = self.sigma
        mu_x = torch.zeros(dim_x, device=x.device, dtype=x.dtype)
        sigma_x = torch.eye(dim_x, device=x.device, dtype=x.dtype) / (sigma**2)
        px = torch.distributions.MultivariateNormal(mu_x, sigma_x)
        self.w = px.sample((self.rff_dim,))
        p = torch.distributions.uniform.Uniform(
            torch.tensor([0.0], device=x.device, dtype=x.dtype),
            2 * torch.tensor([math.pi], device=x.device, dtype=x.dtype),
        )
        self.b = p.sample((self.rff_dim,)).squeeze(1)

    def __call__(self, x):
        if self.w is None or self.b is None:
            self._calc_w_b(x)

        x = x.to(dtype=self.w.dtype)
        device = x.device

        if len(x.shape) == 1:
            x = x.unsqueeze(-1)

        phi_x = math.sqrt(2 / self.rff_dim) * torch.cos(
            torch.mm(x, self.w.to(device=device).t()) + self.b.to(device=device)
        )

        return phi_x.to(dtype=self.w.dtype)
