import torch
import torch.nn as nn
from .BaseModel import BaseModel


class DistMult(BaseModel):
    """
    DistMult: Embedding Entities and Relations for Learning and Inference in KGs.
    Yang et al., ICLR Workshop 2015.
    Score: <h, r, t> = sum(h * r * t)
    """
    def __init__(self, config):
        super(DistMult, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        self.input_drop = nn.Dropout(kwargs.get('input_dropout', 0.2)).to(self.device)

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)          # (B, d)
        r = self.R(batch_r)          # (B, d)
        # DistMult is symmetric: inverse is handled by relation sign
        # For inverse prediction, we use the same relation (symmetric assumption)
        hr = h * r                   # (B, d)
        hr = self.input_drop(hr)

        # 1-N scoring: score(h,r,e) = sum(h*r*e) = (h*r) @ E^T
        scores = torch.mm(hr, self.E.weight.T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)   # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
