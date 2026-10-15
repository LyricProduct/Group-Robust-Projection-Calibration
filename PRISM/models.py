import torch


class embedding_transformer(torch.nn.Module):
    def __init__(self, embed_dim=768, identity=False):
        super(embedding_transformer, self).__init__()
        self.transformer = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        if identity:
            self.transformer.weight.data = torch.eye(embed_dim, dtype=torch.float32)

    def forward(self, x):
        x = self.transformer(x)
        return x


def get_transformer(args, embed_dim):
    if args.init_weight == "i":
        print("Got identity initialization")
        transformer = embedding_transformer(embed_dim=embed_dim, identity=True)
    else:
        transformer = embedding_transformer(embed_dim=embed_dim)

    transformer.to(args.device)
    return transformer
