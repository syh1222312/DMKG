import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel


class InteractE(BaseModel):
    """
    InteractE: Improving Convolution-based Knowledge Graph Embeddings by
    Increasing Feature Interactions.
    Vashishth et al., AAAI 2020.
    Uses feature permutation and circular convolution to increase interactions.
    """
    def __init__(self, config):
        super(InteractE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.reshape = kwargs.get('reshape', [10, 20])
        self.kernel_size = kwargs.get('conv_kernel_size', [9, 11])
        self.out_channels = kwargs.get('conv_out_channels', 96)
        self.num_perm = kwargs.get('num_perm', 1)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        k_h, k_w = self.kernel_size

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)

        self.input_drop = nn.Dropout(kwargs.get('input_dropout', 0.2)).to(self.device)
        self.feature_map_drop = nn.Dropout2d(kwargs.get('feature_map_dropout', 0.2)).to(self.device)
        self.hidden_drop = nn.Dropout(kwargs.get('hidden_dropout', 0.3)).to(self.device)

        self.bn0 = nn.BatchNorm2d(self.num_perm).to(self.device)
        self.bn1 = nn.BatchNorm2d(self.out_channels * self.num_perm).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.emb_dim).to(self.device)

        # Circular padding size
        self.pad_h = k_h // 2
        self.pad_w = k_w // 2

        # Output height/width after circular conv (same padding)
        conv_out_h = 2 * self.reshape[0]
        conv_out_w = self.reshape[1]
        fc_in = self.out_channels * self.num_perm * conv_out_h * conv_out_w

        self.conv = nn.Conv2d(self.num_perm, self.out_channels * self.num_perm,
                              self.kernel_size, padding=0,
                              groups=self.num_perm).to(self.device)
        self.fc = nn.Linear(fc_in, self.emb_dim).to(self.device)
        self.b = Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)

    def circular_padding(self, x, pad_h, pad_w):
        """Circular (wrap-around) padding for 2D feature maps."""
        x = torch.cat([x[..., -pad_w:], x, x[..., :pad_w]], dim=-1)
        x = torch.cat([x[..., -pad_h:, :], x, x[..., :pad_h, :]], dim=-2)
        return x

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)    # (B, d)
        r = self.R(batch_r)    # (B, d)
        if inverse:
            r = -r

        # Interleave h and r to create feature interaction
        hr_interleaved = torch.stack([h, r], dim=2).view(
            -1, 2 * self.emb_dim)                         # (B, 2d) interleaved
        h_part = hr_interleaved[:, 0::2]                   # (B, d) even positions = h
        r_part = hr_interleaved[:, 1::2]                   # (B, d) odd positions = r

        h2d = h_part.view(-1, 1, self.reshape[0], self.reshape[1])
        r2d = r_part.view(-1, 1, self.reshape[0], self.reshape[1])
        stacked = torch.cat([h2d, r2d], dim=2)             # (B, 1, 2H, W)
        stacked = stacked.expand(-1, self.num_perm, -1, -1)
        stacked = self.bn0(stacked)
        stacked = self.input_drop(stacked)

        # Circular convolution
        stacked_pad = self.circular_padding(stacked, self.pad_h, self.pad_w)
        x = self.conv(stacked_pad)                          # (B, C*P, H', W')
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        scores = torch.mm(x, self.E.weight.T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)   # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
