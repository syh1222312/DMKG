import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from .BaseModel import BaseModel


class RGCN(BaseModel):
    """
    RGCN: Modeling Relational Data with Graph Convolutional Networks.
    Schlichtkrull et al., ESWC 2018.
    Uses relational GCN to encode entity representations, then scores via DistMult.
    Note: requires torch_geometric. Install with: pip install torch_geometric
    """
    def __init__(self, config):
        super(RGCN, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.num_layers = kwargs.get('num_layers', 2)
        self.num_bases = kwargs.get('num_bases', 30)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        # Base entity embeddings (input features)
        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)

        # RGCN layers
        self.conv_layers = nn.ModuleList([
            RGCNConv(self.emb_dim, self.emb_dim,
                     num_relations=self.relation_cnt,
                     num_bases=self.num_bases).to(self.device)
            for _ in range(self.num_layers)
        ])
        self.dropout = nn.Dropout(kwargs.get('input_dropout', 0.2)).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')

        # Store edge_index and edge_type from training data
        self.edge_index = None
        self.edge_type = None
        self._build_graph(config.get('data', []))
        self.init()

    def _build_graph(self, train_data):
        """Build the edge_index and edge_type tensors from training triples."""
        if not train_data:
            return
        src, dst, rel = zip(*[(h, t, r) for h, t, r in train_data])
        # Add inverse edges
        inv_src, inv_dst, inv_rel = dst, src, [r + self.relation_cnt for r in rel]
        all_src = list(src) + list(inv_src)
        all_dst = list(dst) + list(inv_dst)
        all_rel = list(rel) + inv_rel

        self.edge_index = torch.tensor([all_src, all_dst], dtype=torch.long).to(self.device)
        self.edge_type = torch.tensor(all_rel, dtype=torch.long).to(self.device)
        # Update num_relations to account for inverses
        for layer in getattr(self, 'conv_layers', []):
            layer.num_relations = self.relation_cnt * 2

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)

    def encode(self):
        """Run RGCN message passing to get contextual entity embeddings."""
        if self.edge_index is None:
            return self.E.weight
        x = self.E.weight
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, self.edge_index, self.edge_type)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        # Encode all entities via RGCN
        E_enc = self.encode()                   # (N, d)
        h = E_enc[batch_h]                      # (B, d)
        r = self.R(batch_r)                     # (B, d)
        if inverse:
            r = -r

        # DistMult decoder
        hr = h * r                              # (B, d)
        hr = self.dropout(hr)
        scores = torch.mm(hr, E_enc.T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)          # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores


class RGCNSimple(BaseModel):
    """
    Simplified RGCN that does NOT require torch_geometric.
    Uses a custom basis-decomposed graph convolution.
    """
    def __init__(self, config):
        super(RGCNSimple, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.num_bases = kwargs.get('num_bases', min(30, self.relation_cnt))
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        num_rel_total = self.relation_cnt * 2  # + inverses

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)

        # Basis decomposition: W_r = sum_b a_{r,b} * V_b
        self.basis = nn.Parameter(
            torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim)).to(self.device)
        self.basis_coef = nn.Parameter(
            torch.Tensor(num_rel_total, self.num_bases)).to(self.device)

        self.dropout = nn.Dropout(kwargs.get('input_dropout', 0.2)).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')

        # Sparse adjacency per relation
        self._adj = {}
        self._build_adj(config.get('data', []))
        self.init()

    def _build_adj(self, train_data):
        """Build sparse adjacency matrices per relation."""
        from collections import defaultdict
        rel_edges = defaultdict(lambda: ([], []))
        for h, t, r in train_data:
            rel_edges[r][0].append(h)
            rel_edges[r][1].append(t)
            # Inverse
            rel_edges[r + self.relation_cnt][0].append(t)
            rel_edges[r + self.relation_cnt][1].append(h)
        self._rel_edges = dict(rel_edges)

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)
        nn.init.xavier_uniform_(self.basis.data)
        nn.init.xavier_uniform_(self.basis_coef.data)

    def encode(self):
        """One-layer RGCN with basis decomposition (simplified, no sparse ops)."""
        x = self.E.weight    # (N, d)
        # W_r = sum_b a_{r,b} * basis[b]  -> (num_rel, d, d)
        W = torch.einsum('rb,bde->rde', self.basis_coef, self.basis)  # (R, d, d)
        out = torch.zeros_like(x)
        for r_id, (src_list, dst_list) in self._rel_edges.items():
            if not src_list:
                continue
            src = torch.tensor(src_list, device=self.device)
            dst = torch.tensor(dst_list, device=self.device)
            # Message: W_r @ x[src]
            msgs = torch.mm(x[src], W[r_id])   # (E_r, d)
            # Aggregate to dst (mean)
            out.index_add_(0, dst, msgs)
        # Normalize by degree (approximation)
        deg = torch.tensor(
            [len(self._rel_edges.get(r, ([],[]))[1]) for r in range(self.relation_cnt * 2)],
            dtype=torch.float32, device=self.device
        ).sum().clamp(min=1)
        out = F.relu(out / (deg / self.entity_cnt))
        return out

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        E_enc = self.encode()
        h = E_enc[batch_h]
        r = self.R(batch_r)
        if inverse:
            r = -r
        hr = h * r
        hr = self.dropout(hr)
        scores = torch.mm(hr, E_enc.T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
