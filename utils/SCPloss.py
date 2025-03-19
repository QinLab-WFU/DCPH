import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class Criterion(torch.nn.Module):
    def __init__(self, embed_dim, n_classes, device):
        super(Criterion, self).__init__()
        self.device = device

        self.in_features = embed_dim
        self.out_features = n_classes

        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features).to(device))
        nn.init.xavier_uniform_(self.weight)

        self.ls_eps = 0

        self.tau = 0.25
        self.psi = 0.25
        self.sp = 1.5
        self.sn = 1
        self.mu = 1.0
        self.b = 2

    def forward(self, batch, labels):
        one_hot = labels.to(self.device)
        cosine = F.linear(F.normalize(batch), F.normalize(self.weight))

        # one_hot = torch.zeros(cosine.size(), device=self.par)
        # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        tp = ((cosine.clamp(min=0.0) * one_hot) * 2).sum() + self.b

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        lossp = ((1.0 - cosine) * torch.exp((1.0 - cosine) * self.sp).detach() * one_hot).sum()

        mask = cosine > self.tau
        cosine = cosine[mask]
        lossn = ((cosine - self.psi)
                 * torch.exp((cosine - self.mu) * self.sn).detach()
                 * (1 - one_hot[mask])).sum()

        loss = (1.0 - (tp) / (tp + lossp + lossn))

        return loss