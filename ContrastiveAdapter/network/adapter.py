import torch.nn as nn


class MultiLayerAdapter(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=512,
        batch_norm=True,
        residual_connection=False,
        queries=None,
        temperature=100.0,
        return_hidden_by_default=True,
        num_encoder_layers=1,
        activation="relu",
    ):
        super().__init__()
        if activation == "relu":
            activation = nn.ReLU(True)
        else:
            batch_norm = False
            activation = nn.GELU()
        self.BatchNorm1d = nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=not batch_norm),
            self.BatchNorm1d,
            activation,
            nn.Linear(hidden_dim, input_dim),
        )
        self.residual_connection = residual_connection

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = input_dim
        self.batch_norm = batch_norm

        self.queries = queries
        self.temperature = temperature

        self.return_hidden_by_default = return_hidden_by_default
        self.num_encoder_layers = num_encoder_layers

    def classifier(self, z):
        z = z / z.norm(dim=-1, keepdim=True)
        return self.temperature * (z @ self.queries.T)

    def forward(self, z, return_hidden=False):
        z = self.encode(z)
        y = self.classifier(z)
        if return_hidden or self.return_hidden_by_default:
            return y, z
        else:
            return y

    def encode(self, z):
        for _ in range(self.num_encoder_layers):
            z = self.encode_single_layer(z)
        return z

    def encode_single_layer(self, z):
        if self.residual_connection:
            return self.encoder(z) + z
        return self.encoder(z)

    def to_device(self, device):
        self.to(device)
        self.queries = self.queries.to(device)
