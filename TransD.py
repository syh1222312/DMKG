import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class TransD(BaseModel):
    """
    TransD: Knowledge Graph Embedding via Dynamic Mapping Matrix.
    Ji et al., ACL 2015.
    Builds dynamic projection matrices from entity and relation projection vectors.
    """
    def __init__(self, config):
        super(TransD, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('emb_dim', 200)
        self.rel_dim = kwargs.get('rel_dim', self.ent_dim)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt, self.ent_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.rel_dim).to(self.device)
        # Projection vectors (one per entity and one per relation)
        self.Ep = nn.Embedding(self.entity_cnt, self.ent_dim).to(self.device)
        self.Rp = nn.Embedding(self.relation_cnt, self.rel_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)
        nn.init.xavier_uniform_(self.Ep.weight.data)
        nn.init.xavier_uniform_(self.Rp.weight.data)
        self.E.weight.data = F.normalize(self.E.weight.data, p=2, dim=1)
        self.R.weight.data = F.normalize(self.R.weight.data, p=2, dim=1)

    def dynamic_project(self, e, ep, rp):
        """
        Dynamic mapping: M_rh = rp^T ep + I_min
        Project e: e_proj = M * e = (rp·ep)e + e[:rel_dim] truncated
        Simplified: e_proj = e[:rel_dim] + (ep·e) * rp
        """
        # e: (..., ent_dim), ep: (B, ent_dim), rp: (B, rel_dim)
        dot = (ep * e).sum(dim=-1, keepdim=True)     # (B, 1) or broadcast
        proj = e[..., :self.rel_dim] + dot * rp       # (..., rel_dim)
        return F.normalize(proj, p=2, dim=-1)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)            # (B, ent_dim)
        r = self.R(batch_r)            # (B, rel_dim)
        hp = self.Ep(batch_h)          # (B, ent_dim)
        rp = self.Rp(batch_r)          # (B, rel_dim)
        if inverse:
            r = -r

        h_proj = self.dynamic_project(h, hp, rp)    # (B, rel_dim)
        query = h_proj + r                           # (B, rel_dim)

        # Score all entities
        all_e = self.E.weight                        # (N, ent_dim)
        all_ep = self.Ep.weight                      # (N, ent_dim)
        B = batch_h.size(0)

        # For each (b, n): proj = all_e[n,:rel_dim] + (all_ep[n]·all_e[n]) * rp[b]
        # dot_en: (N,) for each entity's self-projection scalar
        dot_en = (all_ep * all_e).sum(dim=-1)        # (N,)
        e_base = all_e[:, :self.rel_dim]             # (N, rel_dim)

        # all_e_proj[b, n, :] = e_base[n] + dot_en[n] * rp[b]
        # (1, N, rel_dim) + (N,) * (B, 1, rel_dim) -> (B, N, rel_dim)
        rp_exp = rp.unsqueeze(1)                                   # (B, 1, rel_dim)
        dot_exp = dot_en.unsqueeze(0).unsqueeze(-1)                # (1, N, 1)
        all_e_proj = e_base.unsqueeze(0) + dot_exp * rp_exp       # (B, N, rel_dim)
        all_e_proj = F.normalize(all_e_proj, p=2, dim=-1)

        q_exp = query.unsqueeze(1)                                  # (B, 1, rel_dim)
        dist = (q_exp - all_e_proj).norm(p=2, dim=-1)              # (B, N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
