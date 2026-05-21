import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class KGCN(BaseModel):
    """
    KGCN: Knowledge Graph Convolutional Networks in the Game of Thrones.
    Wang et al., WWW 2019.
    Applies graph convolution aggregating neighborhood information with
    relation-attention weighting for knowledge graph link prediction.
    Adapted from recommendation setting to standard link prediction.
    """
    def __init__(self, config):
        super(KGCN, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 64)
        self.n_iter = kwargs.get('n_iter', 2)
        self.neighbor_sample_size = kwargs.get('neighbor_sample_size', 8)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt + 1, self.emb_dim,
                              padding_idx=self.entity_cnt).to(self.device)
        self.R = nn.Embedding(self.relation_cnt + 1, self.emb_dim,
                              padding_idx=self.relation_cnt).to(self.device)
        self.W = nn.ModuleList([
            nn.Linear(self.emb_dim, self.emb_dim, bias=False).to(self.device)
            for _ in range(self.n_iter)
        ])
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.dropout = nn.Dropout(kwargs.get('input_dropout', 0.5)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')

        # Build neighbor table from training data
        self.neighbor_table = self._build_neighbor_table(
            config.get('data', []),
            self.neighbor_sample_size
        )
        self.init()

    def _build_neighbor_table(self, train_data, K):
        """
        Build neighbor table: for each entity, sample up to K (neighbor, relation) pairs.
        Returns neighbor_ent: (N+1, K) and neighbor_rel: (N+1, K) tensors.
        """
        from collections import defaultdict
        adj = defaultdict(list)
        for h, t, r in train_data:
            adj[h].append((t, r))
            adj[t].append((h, r))

        N = self.entity_cnt
        neighbor_ent = torch.full((N + 1, K), N, dtype=torch.long)    # padding = entity_cnt
        neighbor_rel = torch.full((N + 1, K), self.relation_cnt, dtype=torch.long)

        for ent_id in range(N):
            neighbors = adj[ent_id]
            if not neighbors:
                continue
            # Sample K neighbors (with replacement if needed)
            indices = torch.randint(len(neighbors), (K,))
            for k, idx in enumerate(indices):
                neighbor_ent[ent_id, k] = neighbors[idx][0]
                neighbor_rel[ent_id, k] = neighbors[idx][1]

        return {
            'ent': neighbor_ent.to(self.device),
            'rel': neighbor_rel.to(self.device)
        }

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)

    def aggregate(self, entity_ids, relation_embed):
        """
        Aggregate neighborhood for entity_ids using relation-attention.
        entity_ids: (B,) entity indices
        relation_embed: (B, d) relation embeddings for attention
        Returns aggregated embeddings: (B, d)
        """
        # Neighbor lookup
        nb_ent = self.neighbor_table['ent'][entity_ids]   # (B, K)
        nb_rel = self.neighbor_table['rel'][entity_ids]   # (B, K)

        # Embeddings
        nb_ent_emb = self.E(nb_ent)    # (B, K, d)
        nb_rel_emb = self.R(nb_rel)    # (B, K, d)

        # Relation-attention: score each neighbor by its relation similarity to query r
        rel_q = relation_embed.unsqueeze(1)                 # (B, 1, d)
        attn_scores = (nb_rel_emb * rel_q).sum(dim=-1)      # (B, K)
        attn_weights = F.softmax(attn_scores, dim=-1)        # (B, K)

        # Weighted aggregation
        agg = (attn_weights.unsqueeze(-1) * nb_ent_emb).sum(dim=1)  # (B, d)
        return agg

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        r_emb = self.R(batch_r)                             # (B, d)
        if inverse:
            r_emb = -r_emb

        # Multi-hop aggregation for head entities
        entity_repr = self.E(batch_h)                       # (B, d)
        for i in range(self.n_iter):
            agg = self.aggregate(batch_h, r_emb)            # (B, d)
            entity_repr = F.relu(self.W[i](entity_repr + agg))
            entity_repr = self.dropout(entity_repr)

        # Score against all entities
        scores = torch.mm(entity_repr * r_emb,
                          self.E.weight[:self.entity_cnt].T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)   # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
