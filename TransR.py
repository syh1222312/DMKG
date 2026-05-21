import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class TransR(BaseModel):
    """
    TransR: Learning Entity and Relation Embeddings for Knowledge Graph Completion.
    Lin et al., AAAI 2015.
    Projects entities into relation-specific spaces via projection matrices.
    """
    def __init__(self, config):
        super(TransR, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.ent_dim = kwargs.get('emb_dim', 200)
        self.rel_dim = kwargs.get('rel_dim', 200)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt, self.ent_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.rel_dim).to(self.device)
        # Projection matrix: maps entity space -> relation space
        self.M = nn.Embedding(self.relation_cnt, self.ent_dim * self.rel_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)
        # Initialize projection matrix as identity-like
        eye = torch.eye(min(self.ent_dim, self.rel_dim))
        M_init = torch.zeros(self.ent_dim, self.rel_dim)
        M_init[:min(self.ent_dim, self.rel_dim), :min(self.ent_dim, self.rel_dim)] = eye
        M_flat = M_init.view(-1).unsqueeze(0).expand(self.relation_cnt, -1)
        self.M.weight.data = M_flat.clone()

    def project(self, e, M_flat):
        """
        e: (B, ent_dim) or (N, ent_dim)
        M_flat: (B, ent_dim * rel_dim)
        Returns: projected (B, rel_dim)
        """
        B = M_flat.size(0)
        M = M_flat.view(B, self.ent_dim, self.rel_dim)          # (B, ent_dim, rel_dim)
        e = e.unsqueeze(1)                                        # (B, 1, ent_dim) or needs broadcast
        out = torch.bmm(e, M).squeeze(1)                         # (B, rel_dim)
        return F.normalize(out, p=2, dim=-1)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)          # (B, ent_dim)
        r = self.R(batch_r)          # (B, rel_dim)
        M_flat = self.M(batch_r)     # (B, ent_dim * rel_dim)
        if inverse:
            r = -r

        h_proj = self.project(h, M_flat)     # (B, rel_dim)
        query = h_proj + r                    # (B, rel_dim)

        # Score all entities (project each entity embedding)
        all_e = self.E.weight          # (N, ent_dim)
        B = batch_h.size(0)
        M = M_flat.view(B, self.ent_dim, self.rel_dim)   # (B, ent_dim, rel_dim)

        # all_e_proj[b, n] = all_e[n] @ M[b]
        # (B, N, ent_dim) x (B, ent_dim, rel_dim) -> (B, N, rel_dim)
        e_exp = all_e.unsqueeze(0).expand(B, -1, -1)            # (B, N, ent_dim)
        all_e_proj = torch.bmm(e_exp, M)                         # (B, N, rel_dim)
        all_e_proj = F.normalize(all_e_proj, p=2, dim=-1)

        q_exp = query.unsqueeze(1)                                # (B, 1, rel_dim)
        dist = (q_exp - all_e_proj).norm(p=2, dim=-1)            # (B, N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
