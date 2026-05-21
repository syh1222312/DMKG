import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class TransH(BaseModel):
    """
    TransH: Knowledge Graph Embedding by Translating on Hyperplanes.
    Wang et al., AAAI 2014.
    Projects entities onto relation-specific hyperplanes before translation.
    """
    def __init__(self, config):
        super(TransH, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        # Normal vectors to the relation hyperplanes
        self.W = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)
        nn.init.xavier_uniform_(self.W.weight.data)
        self.E.weight.data = F.normalize(self.E.weight.data, p=2, dim=1)
        self.W.weight.data = F.normalize(self.W.weight.data, p=2, dim=1)

    def project_to_hyperplane(self, e, w_norm):
        """Project entity embedding e onto the hyperplane with normal w_norm."""
        # e_proj = e - (e·w)w
        return e - (e * w_norm).sum(dim=-1, keepdim=True) * w_norm

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)                                   # (B, d)
        r = self.R(batch_r)                                   # (B, d)
        w = F.normalize(self.W(batch_r), p=2, dim=1)          # (B, d)
        if inverse:
            r = -r

        h_proj = self.project_to_hyperplane(h, w)             # (B, d)
        query = h_proj + r                                     # (B, d)

        # Score against ALL entities after projecting them
        all_e = self.E.weight                                  # (N, d)
        # Project all entities: e_proj = e - (e·w)w, broadcast over batch
        # w: (B, 1, d), all_e: (1, N, d)
        w_exp = w.unsqueeze(1)                                  # (B, 1, d)
        e_exp = all_e.unsqueeze(0)                              # (1, N, d)
        dot = (e_exp * w_exp).sum(dim=-1, keepdim=True)        # (B, N, 1)
        all_e_proj = e_exp - dot * w_exp                       # (B, N, d)

        q_exp = query.unsqueeze(1)                              # (B, 1, d)
        diff = q_exp - all_e_proj                               # (B, N, d)
        dist = diff.norm(p=2, dim=-1)                           # (B, N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
