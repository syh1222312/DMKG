import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class RotatE(BaseModel):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.
    Sun et al., ICLR 2019.
    Entities: complex vectors; Relations: unit complex rotations.
    Score: -||h ∘ r - t||  (complex element-wise product).
    """
    def __init__(self, config):
        super(RotatE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)   # complex dim = emb_dim/2 complex numbers
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        self.gamma = kwargs.get('gamma', 12.0)
        self.epsilon = 2.0

        # Entity embeddings (real + imag interleaved, or separate)
        self.E_re = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.E_im = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        # Relation embeddings as phase angles (mapped to unit circle)
        self.R_phase = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E_re.weight.data)
        nn.init.xavier_uniform_(self.E_im.weight.data)
        nn.init.uniform_(self.R_phase.weight.data, -3.14159, 3.14159)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h_re = self.E_re(batch_h)      # (B, d)
        h_im = self.E_im(batch_h)
        phase = self.R_phase(batch_r)  # (B, d) — phase angle
        if inverse:
            phase = -phase

        # Unit rotation: r = e^{i*phase}
        r_re = torch.cos(phase)        # (B, d)
        r_im = torch.sin(phase)

        # h ∘ r (complex multiplication):
        # (h_re + i*h_im)(r_re + i*r_im) = (h_re*r_re - h_im*r_im) + i*(h_re*r_im + h_im*r_re)
        hr_re = h_re * r_re - h_im * r_im    # (B, d)
        hr_im = h_re * r_im + h_im * r_re

        # Score against all entities: -||hr - t||  (L2 in complex space)
        # ||hr - t||^2 = ||hr_re - t_re||^2 + ||hr_im - t_im||^2
        all_e_re = self.E_re.weight    # (N, d)
        all_e_im = self.E_im.weight

        # Efficient distance computation
        diff_re = hr_re.unsqueeze(1) - all_e_re.unsqueeze(0)   # (B, N, d)
        diff_im = hr_im.unsqueeze(1) - all_e_im.unsqueeze(0)
        dist = (diff_re ** 2 + diff_im ** 2).sum(dim=-1).sqrt()  # (B, N)

        scores = torch.sigmoid(-dist + self.b.unsqueeze(0) + self.gamma)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
