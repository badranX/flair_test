import torch
from torch import nn
from .config import Config


class Encoder(nn.Module):
    def __init__(self, hp=Config()):
        super().__init__()
        self.e = nn.Embedding(hp.k_neck, hp.model_d)
        self.encoder = nn.Sequential(
                    nn.Linear(hp.input_d, 2*hp.model_d),
                    nn.Sigmoid(),
                    nn.Linear(2*hp.model_d, hp.model_d))

        #self.embed = nn.Linear(hp.model_d, k*100, bias=False)
        self.embed = nn.Linear(hp.model_d, hp.k_neck, bias=False)
        self.fake_embed = nn.Linear(hp.model_d, hp.model_d, bias=False)

        self.probs = nn.Softmax(dim=-1)
        self.saved_probs = None
        self.saved_perplexity = None

    def perplixity(self, win_indices):
        prep = torch.bincount(win_indices.view(-1))
        size = torch.numel(prep)
        prep = prep/torch.numel(prep)
        prep = torch.exp(-torch.sum(prep* torch.log(prep + 1e+10)))
        return prep


    def forward(self, x):
        shape = x.shape
        x = x.view(-1, x.shape[-1])

        x = self.encoder(x)
        use_embedding = False
        if use_embedding:
            x = self.embed(x) #logits
        else:
            x = self.fake_embed(x)

        #prepare a sample, not part of gradient tree
        probs = self.probs(x)
        self.saved_probs = probs
        if use_embedding:
            ind = torch.argmax(probs, dim=-1)
            self.saved_perplexity = self.perplixity(ind)
            if True:
                probs_shape = probs.size()
                hard = torch.zeros_like(probs).view(-1, probs_shape[-1])
                hard.scatter_(1, ind.view(-1, 1), 1)
                hard = (hard - probs).detach() + probs

                x = hard@self.e.weight
            else:
                win = self.e(ind)
                soft_win = probs@self.e.weight

                x = (win - soft_win).detach() + soft_win
        
        elif False:
            x = probs@self.e.weight
            #x = x

        
        #winners = torch.zeros_like(x).scatter_(1, winners.unsqueeze(1), 1.)
        #q = (embeddings - x).detach() + x
        x = x.view(shape[0], shape[1], -1)
        return x



class Decoder(nn.Module):
    def __init__(self, hp=Config()):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(hp.model_size, hp.model_d))

        self.mlt = nn.MultiheadAttention(hp.model_d, hp.num_heads)
        self.mlt2 = nn.MultiheadAttention(hp.model_d, hp.num_heads)
        self.l = nn.Sequential(
                    nn.Linear(hp.model_d, 2*hp.model_d),
                    nn.Sigmoid(),
                    nn.Linear(2*hp.model_d, hp.model_d))
        self.mlp = nn.Sequential(
                    nn.Linear(hp.model_d, hp.model_d),
                    nn.Sigmoid(),
                    nn.Linear(hp.model_d, hp.k),
                    )
        self.norm = nn.LayerNorm(hp.model_d)
        #self.probs = nn.Softmax(dim=-1)

    def pos_encoding(self, x):
        return x + self.pos[:x.shape[-2],:]


    def forward(self, x):
        x = self.pos_encoding(x)
        #x = self.l(x)

        z, attn_output_weights = self.mlt(x, x, x)
        x = x + self.norm(z)
        z, attn_output_weights = self.mlt(x, x, x)
        x = x + self.norm(z)
        x = self.mlp(x)
        #probs = self.probs(x)
        return x


class VQbottleneck(nn.Module):
    def __init__(self, hp=Config()):
        super().__init__()
        self.encoder = Encoder(hp)
        self.decoder = Decoder(hp)

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)

        return y
