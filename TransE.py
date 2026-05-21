import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class TransE(BaseModel):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data.
    Bordes et al., NeurIPS 2013.
    Score: -||h + r - t||_2
    """
    def __init__(self, config):
        super(TransE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.norm = kwargs.get('norm', 2)
        self.margin = kwargs.get('margin', 1.0)
        label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)
        # Normalize entity embeddings
        self.E.weight.data = F.normalize(self.E.weight.data, p=2, dim=1)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)          # (B, d)
        r = self.R(batch_r)          # (B, d)
        if inverse:
            r = -r
        query = h + r                # (B, d)

        # Score against ALL entities (1-N scoring)
        # score(h,r,e) = sigmoid(- ||h+r - e||_2)
        all_e = self.E.weight         # (N, d)
        # Efficient squared distance: ||query - e||^2 = ||q||^2 - 2*q*e^T + ||e||^2
        q2 = (query ** 2).sum(dim=1, keepdim=True)         # (B, 1)
        e2 = (all_e ** 2).sum(dim=1, keepdim=True).T       # (1, N)
        qe = torch.mm(query, all_e.T)                       # (B, N)
        dist2 = (q2 + e2 - 2 * qe).clamp(min=1e-8)
        dist = dist2.sqrt()
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))  # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
