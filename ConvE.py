import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel


class ConvE(BaseModel):
    """
    ConvE: Convolutional 2D Knowledge Graph Embeddings.
    Dettmers et al., AAAI 2018.
    Applies 2D convolution on reshaped and concatenated h, r embeddings.
    """
    def __init__(self, config):
        super(ConvE, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)
        self.reshape = kwargs.get('reshape', [10, 20])
        self.kernel_size = kwargs.get('conv_kernel_size', [3, 3])
        self.out_channels = kwargs.get('conv_out_channels', 32)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)

        self.E = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)

        self.input_drop = nn.Dropout(kwargs.get('input_dropout', 0.2)).to(self.device)
        self.feature_map_drop = nn.Dropout2d(kwargs.get('feature_map_dropout', 0.2)).to(self.device)
        self.hidden_drop = nn.Dropout(kwargs.get('hidden_dropout', 0.3)).to(self.device)

        self.bn0 = nn.BatchNorm2d(1).to(self.device)
        self.bn1 = nn.BatchNorm2d(self.out_channels).to(self.device)
        self.bn2 = nn.BatchNorm1d(self.emb_dim).to(self.device)

        # Input to conv: concatenate h and r along rows -> (2*reshape[0], reshape[1])
        conv_h = 2 * self.reshape[0] - self.kernel_size[0] + 1
        conv_w = self.reshape[1] - self.kernel_size[1] + 1
        fc_in = self.out_channels * conv_h * conv_w

        self.conv = nn.Conv2d(1, self.out_channels, self.kernel_size).to(self.device)
        self.fc = nn.Linear(fc_in, self.emb_dim).to(self.device)
        self.b = Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E.weight.data)
        nn.init.xavier_uniform_(self.R.weight.data)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h = self.E(batch_h)    # (B, d)
        r = self.R(batch_r)    # (B, d)
        if inverse:
            r = -r

        # Reshape and concatenate
        h2d = h.view(-1, 1, self.reshape[0], self.reshape[1])
        r2d = r.view(-1, 1, self.reshape[0], self.reshape[1])
        stacked = torch.cat([h2d, r2d], 2)                      # (B, 1, 2*H, W)
        stacked = self.bn0(stacked)
        stacked = self.input_drop(stacked)

        x = self.conv(stacked)                                   # (B, C, H', W')
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        scores = torch.mm(x, self.E.weight.T) + self.b.unsqueeze(0)
        scores = torch.sigmoid(scores)    # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
