import torch
import torch.nn as nn
from .BaseModel import BaseModel


class ComplEx(BaseModel):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction.
    Trouillon et al., ICML 2016.
    Uses complex-valued embeddings; score = Re(<h, r, conj(t)>).
    """
    def __init__(self, config):
        super(ComplEx, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.emb_dim = kwargs.get('emb_dim', 200)   # each embedding has dim/2 real + dim/2 imag
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
        self.input_drop = nn.Dropout(kwargs.get('input_dropout', 0.2)).to(self.device)

        # Store real and imaginary parts separately
        self.E_re = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.E_im = nn.Embedding(self.entity_cnt, self.emb_dim).to(self.device)
        self.R_re = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        self.R_im = nn.Embedding(self.relation_cnt, self.emb_dim).to(self.device)
        self.b = nn.Parameter(torch.zeros(self.entity_cnt)).to(self.device)
        self.loss_fn = nn.BCELoss(reduction='sum')
        self.init()

    def init(self):
        nn.init.xavier_uniform_(self.E_re.weight.data)
        nn.init.xavier_uniform_(self.E_im.weight.data)
        nn.init.xavier_uniform_(self.R_re.weight.data)
        nn.init.xavier_uniform_(self.R_im.weight.data)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        h_re = self.E_re(batch_h)    # (B, d)
        h_im = self.E_im(batch_h)
        r_re = self.R_re(batch_r)
        r_im = self.R_im(batch_r)
        if inverse:
            # Conjugate relation for inverse
            r_im = -r_im

        h_re = self.input_drop(h_re)
        h_im = self.input_drop(h_im)

        # Re(<h, r, conj(t)>) = h_re*r_re*t_re + h_re*r_im*t_im
        #                      + h_im*r_re*t_im - h_im*r_im*t_re
        # = (h_re*r_re + h_im*r_im) @ t_re^T + (h_re*r_im - h_im*r_re) @ t_im^T
        A = h_re * r_re + h_im * r_im   # (B, d)
        B = h_re * r_im - h_im * r_re   # (B, d)

        scores = (torch.mm(A, self.E_re.weight.T)
                  + torch.mm(B, self.E_im.weight.T)
                  + self.b.unsqueeze(0))
        scores = torch.sigmoid(scores)   # (B, N)

        loss = None
        if batch_t is not None:
            batch_size = batch_h.size(0)
            target = torch.zeros(batch_size, self.entity_cnt).to(self.device)
            target.scatter_(1, batch_t.view(-1, 1), 1.0)
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / self.entity_cnt
            loss = self.loss_fn(scores, target) / batch_size
        return loss, scores
