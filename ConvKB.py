import torch
import torch.nn as nn
import torch.nn.functional as F
from .BaseModel import BaseModel


class ConvKB(BaseModel):
    """
    ConvKB: A Novel Embedding Model for Knowledge Base Completion Based on
    Convolutional Neural Network.
    Nguyen et al., NAACL-HLT 2018.
    Applies 1D convolution on the triple (h, r, t) represented as a matrix.
    Adapted for 1-N scoring via score decomposition.
    """
    def __init__(self, config):
        super(ConvKB, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.out_channels = kwargs.get('conv_out_channels', 50)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        # 1D convolution over [h, r, t] as a (3, d) input
        # We split the scoring into a learned interaction over (h, r) then dot with all t
        self.conv1 = nn.Conv1d(3, self.out_channels, kernel_size=1).to(self.device)
        self.fc = nn.Linear(self.emb_dim * self.out_channels, self.emb_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.drop = nn.Dropout(kwargs.get('input_dropout', 0.5)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)     # (B, d)
        r = self.R(batch_r)     # (B, d)
        if inverse:
            r = -r

        # For 1-N scoring: we can compute a context vector from (h, r) that captures
        # the interaction, then score against all entity embeddings via dot product.
        # Approximate 1-N: stack [h, r, zeros] and convolve, then dot with E
        B = batch_h.size(0)
        zero = torch.zeros_like(h)
        triple = torch.stack([h, r, zero], dim=1)           # (B, 3, d)
        triple = self.drop(triple)
        conv_out = F.relu(self.conv1(triple))               # (B, C, d)
        conv_out = conv_out.view(B, -1)                     # (B, C*d)
        context = self.fc(conv_out)                         # (B, d)
        context = self.drop(context)

        scores = torch.mm(context, self.E.weight.T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)   # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
