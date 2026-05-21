"""
run_baselines.py  ——  DMCNNS 知识图谱补全基线实验脚本（自包含版）
============================================================
用法（在 DMKG 项目根目录下运行）：
    python run_baselines.py --dataset FB15K237 --model TransE --epochs 200
    python run_baselines.py --dataset WN18RR   --model RotatE --epochs 200
    python run_baselines.py --dataset FB15K237 --model all    --epochs 200

支持模型: TransE TransH TransR TransD DistMult ComplEx RotatE
          ConvE ConvKB InteractE KGCN
依赖: torch numpy tqdm（均已在 requirements.txt 中）
"""

import os, sys, time, json, argparse, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from math import sqrt

# ── 只复用 DMKG 的 Dataset（数据加载），训练与评估自己实现 ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import Dataset


def _train_epoch(loader, model, optimizer, device):
    """内联训练循环，不依赖项目 train.py 的签名"""
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False):
        h = batch[0].to(device)
        t = batch[1].to(device)
        r = batch[2].to(device)
        # 正向 & 反向各做一次
        for inv in [True, False]:
            optimizer.zero_grad()
            bh, bt = (t, h) if inv else (h, t)
            loss, _ = model(bh, r, bt, inverse=inv)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return losses


def _eval_tail(loader, model, device, data):
    """内联评估函数，不依赖项目 evaluation.py 的签名"""
    hits = [[] for _ in range(10)]
    ranks_right = []
    ent_t = data['entity_relation']['as_tail']
    ent_h = data['entity_relation']['as_head']

    for batch in tqdm(loader, leave=False):
        eh = batch[0].to(device)
        et = batch[1].to(device)
        er = batch[2].to(device)
        _, pred  = model(eh, er)
        _, pred1 = model(et, er, inverse=True)

        for i in range(eh.size(0)):
            ft = ent_t[eh[i].item()][er[i].item()]
            fh = ent_h[et[i].item()][er[i].item()]

            pv  = pred[i][et[i].item()].item()
            pv1 = pred1[i][eh[i].item()].item()
            pred[i][ft]  = 0.0;  pred[i][et[i].item()]  = pv
            pred1[i][fh] = 0.0;  pred1[i][eh[i].item()] = pv1

        _, idx  = torch.sort(pred,  1, descending=True)
        _, idx1 = torch.sort(pred1, 1, descending=True)
        idx  = idx.cpu().numpy()
        idx1 = idx1.cpu().numpy()

        for i in range(eh.size(0)):
            rank  = int(np.where(idx[i]  == et[i].item())[0][0])
            rank1 = int(np.where(idx1[i] == eh[i].item())[0][0])
            ranks_right.append(rank + 1)
            for k in range(10):
                hits[k].append(1.0 if rank <= k else 0.0)
            # (rank1 只用于双向评估，这里只保留尾实体方向)

    # 构造与原 eval_for_tail 相同的返回格式 [hits, _, ranks, _, ranks_right]
    return hits, [], list(ranks_right), [], ranks_right

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ═══════════════════════════════════════════════════════════════════════════════
#  基线模型实现（所有模型均使用与 ConvD 相同的 forward 接口）
#  forward(batch_h, batch_r, batch_t=None, inverse=False) -> (loss, scores)
#  scores: (B, entity_cnt),  sigmoid 输出，越大越好
# ═══════════════════════════════════════════════════════════════════════════════

class BaseKGE(nn.Module):
    def __init__(self, config): super().__init__(); self.config = config
    @classmethod
    def init_model(cls, config): return cls(config)
    def _bce_loss(self, scores, batch_t, entity_cnt, device, label_smoothing=0.1):
        B = scores.size(0)
        target = torch.zeros(B, entity_cnt).to(device)
        target.scatter_(1, batch_t.view(-1,1), 1.0)
        target = (1-label_smoothing)*target + label_smoothing/entity_cnt
        loss = F.binary_cross_entropy(scores, target, reduction='sum') / B
        return loss


# ─── TransE ──────────────────────────────────────────────────────────────────
class TransE(BaseKGE):
    """Bordes et al., NeurIPS 2013. Score: sigmoid(-||h+r-t||)"""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R, d = config['entity_cnt'], config['relation_cnt'], p.get('emb_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.E = nn.Embedding(N, d).to(self.dev)
        self.R = nn.Embedding(R, d).to(self.dev)
        self.b = nn.Parameter(torch.zeros(N)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)
        self.E.weight.data = F.normalize(self.E.weight.data, p=2, dim=1)

    def forward(self, bh, br, bt=None, inverse=False):
        h = self.E(bh); r = self.R(br)
        if inverse: r = -r
        q = h + r                                          # (B,d)
        q2 = (q**2).sum(1,keepdim=True)                   # (B,1)
        e2 = (self.E.weight**2).sum(1,keepdim=True).T     # (1,N)
        dist = (q2 + e2 - 2*torch.mm(q, self.E.weight.T)).clamp(1e-8).sqrt()
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── TransH ──────────────────────────────────────────────────────────────────
class TransH(BaseKGE):
    """Wang et al., AAAI 2014. Translation on relation hyperplanes."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R, d = config['entity_cnt'], config['relation_cnt'], p.get('emb_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.E = nn.Embedding(N, d).to(self.dev)
        self.R = nn.Embedding(R, d).to(self.dev)
        self.W = nn.Embedding(R, d).to(self.dev)   # hyperplane normals
        self.b = nn.Parameter(torch.zeros(N)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)
        nn.init.xavier_uniform_(self.W.weight)
        self.E.weight.data = F.normalize(self.E.weight.data, p=2, dim=1)
        self.W.weight.data = F.normalize(self.W.weight.data, p=2, dim=1)

    def forward(self, bh, br, bt=None, inverse=False):
        h = self.E(bh); r = self.R(br)
        w = F.normalize(self.W(br), p=2, dim=1)
        if inverse: r = -r
        h_proj = h - (h*w).sum(-1,keepdim=True)*w
        q = h_proj + r                                              # (B,d)
        all_e = self.E.weight                                       # (N,d)
        w_e = w.unsqueeze(1)                                        # (B,1,d)
        e_e = all_e.unsqueeze(0)                                    # (1,N,d)
        e_proj = e_e - (e_e*w_e).sum(-1,keepdim=True)*w_e          # (B,N,d)
        dist = (q.unsqueeze(1) - e_proj).norm(p=2,dim=-1)          # (B,N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── TransR ──────────────────────────────────────────────────────────────────
class TransR(BaseKGE):
    """Lin et al., AAAI 2015. Relation-specific projection spaces."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R = config['entity_cnt'], config['relation_cnt']
        de, dr = p.get('emb_dim',200), p.get('rel_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.de = de; self.dr = dr
        self.E = nn.Embedding(N, de).to(self.dev)
        self.R = nn.Embedding(R, dr).to(self.dev)
        self.M = nn.Embedding(R, de*dr).to(self.dev)   # projection matrices
        self.b = nn.Parameter(torch.zeros(N)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)
        # init M as near-identity
        eye = torch.zeros(de, dr); eye[:min(de,dr),:min(de,dr)] = torch.eye(min(de,dr))
        self.M.weight.data = eye.view(-1).unsqueeze(0).expand(R,-1).clone()

    def forward(self, bh, br, bt=None, inverse=False):
        B = bh.size(0)
        h = self.E(bh); r = self.R(br)
        if inverse: r = -r
        M = self.M(br).view(B, self.de, self.dr)          # (B,de,dr)
        h_p = F.normalize(torch.bmm(h.unsqueeze(1), M).squeeze(1), p=2, dim=-1)
        q = h_p + r                                        # (B,dr)
        all_e = self.E.weight                              # (N,de)
        e_p = F.normalize(
            torch.bmm(all_e.unsqueeze(0).expand(B,-1,-1), M),  # (B,N,dr)
            p=2, dim=-1)
        dist = (q.unsqueeze(1) - e_p).norm(p=2, dim=-1)   # (B,N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── TransD ──────────────────────────────────────────────────────────────────
class TransD(BaseKGE):
    """Ji et al., ACL 2015. Dynamic mapping matrices via projection vectors."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R = config['entity_cnt'], config['relation_cnt']
        de, dr = p.get('emb_dim',200), p.get('rel_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.de = de; self.dr = dr
        self.E  = nn.Embedding(N, de).to(self.dev)
        self.R  = nn.Embedding(R, dr).to(self.dev)
        self.Ep = nn.Embedding(N, de).to(self.dev)   # entity proj vectors
        self.Rp = nn.Embedding(R, dr).to(self.dev)   # relation proj vectors
        self.b  = nn.Parameter(torch.zeros(N)).to(self.dev)
        for emb in [self.E,self.R,self.Ep,self.Rp]:
            nn.init.xavier_uniform_(emb.weight)
        self.E.weight.data  = F.normalize(self.E.weight.data,  p=2, dim=1)
        self.R.weight.data  = F.normalize(self.R.weight.data,  p=2, dim=1)

    def forward(self, bh, br, bt=None, inverse=False):
        h = self.E(bh); r = self.R(br); hp = self.Ep(bh); rp = self.Rp(br)
        if inverse: r = -r
        # h_proj = h[:dr] + (hp·h)*rp
        dot_h = (hp*h).sum(-1, keepdim=True)
        h_p = F.normalize(h[...,:self.dr] + dot_h*rp, p=2, dim=-1)
        q = h_p + r                                         # (B,dr)
        # all entity projections: e_proj[n] = E[n,:dr] + (Ep[n]·E[n])*rp[b]
        dot_e = (self.Ep.weight * self.E.weight).sum(-1)   # (N,)
        e_base = self.E.weight[:, :self.dr]                 # (N,dr)
        # broadcast: (1,N,dr) + (N,1)*(B,1,dr)
        e_proj = F.normalize(
            e_base.unsqueeze(0) + dot_e.unsqueeze(0).unsqueeze(-1)*rp.unsqueeze(1),
            p=2, dim=-1)                                    # (B,N,dr)
        dist = (q.unsqueeze(1) - e_proj).norm(p=2, dim=-1) # (B,N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0))
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── DistMult ─────────────────────────────────────────────────────────────────
class DistMult(BaseKGE):
    """Yang et al., ICLR 2015. Score: sigmoid(<h,r,t>)"""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R, d = config['entity_cnt'], config['relation_cnt'], p.get('emb_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.E = nn.Embedding(N, d).to(self.dev)
        self.R = nn.Embedding(R, d).to(self.dev)
        self.b = nn.Parameter(torch.zeros(N)).to(self.dev)
        self.drop = nn.Dropout(p.get('input_dropout',0.2)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)

    def forward(self, bh, br, bt=None, inverse=False):
        hr = self.drop(self.E(bh) * self.R(br))           # (B,d)
        scores = torch.sigmoid(torch.mm(hr, self.E.weight.T) + self.b)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── ComplEx ─────────────────────────────────────────────────────────────────
class ComplEx(BaseKGE):
    """Trouillon et al., ICML 2016. Complex-valued embeddings."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R, d = config['entity_cnt'], config['relation_cnt'], p.get('emb_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.E_re = nn.Embedding(N, d).to(self.dev); self.E_im = nn.Embedding(N, d).to(self.dev)
        self.R_re = nn.Embedding(R, d).to(self.dev); self.R_im = nn.Embedding(R, d).to(self.dev)
        self.b = nn.Parameter(torch.zeros(N)).to(self.dev)
        self.drop = nn.Dropout(p.get('input_dropout',0.2)).to(self.dev)
        for emb in [self.E_re,self.E_im,self.R_re,self.R_im]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, bh, br, bt=None, inverse=False):
        h_re=self.drop(self.E_re(bh)); h_im=self.drop(self.E_im(bh))
        r_re=self.R_re(br); r_im=self.R_im(br)
        if inverse: r_im = -r_im
        # Re(<h,r,conj(t)>) = (h_re*r_re+h_im*r_im)@E_re^T + (h_re*r_im-h_im*r_re)@E_im^T
        A = h_re*r_re + h_im*r_im; B_ = h_re*r_im - h_im*r_re
        scores = torch.sigmoid(
            torch.mm(A, self.E_re.weight.T) + torch.mm(B_, self.E_im.weight.T) + self.b)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── RotatE ──────────────────────────────────────────────────────────────────
class RotatE(BaseKGE):
    """Sun et al., ICLR 2019. Relations as rotations in complex space."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R, d = config['entity_cnt'], config['relation_cnt'], p.get('emb_dim',200)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.gamma = p.get('gamma', 12.0)
        self.E_re = nn.Embedding(N, d).to(self.dev); self.E_im = nn.Embedding(N, d).to(self.dev)
        self.R_ph = nn.Embedding(R, d).to(self.dev)   # phase angles
        self.b = nn.Parameter(torch.zeros(N)).to(self.dev)
        nn.init.xavier_uniform_(self.E_re.weight); nn.init.xavier_uniform_(self.E_im.weight)
        nn.init.uniform_(self.R_ph.weight, -3.14159, 3.14159)

    def forward(self, bh, br, bt=None, inverse=False):
        h_re=self.E_re(bh); h_im=self.E_im(bh)
        ph = self.R_ph(br)
        if inverse: ph = -ph
        r_re=torch.cos(ph); r_im=torch.sin(ph)
        # h ∘ r
        hr_re = h_re*r_re - h_im*r_im; hr_im = h_re*r_im + h_im*r_re
        # distance to all entities
        diff_re = hr_re.unsqueeze(1) - self.E_re.weight.unsqueeze(0)  # (B,N,d)
        diff_im = hr_im.unsqueeze(1) - self.E_im.weight.unsqueeze(0)
        dist = (diff_re**2 + diff_im**2).sum(-1).sqrt()               # (B,N)
        scores = torch.sigmoid(-dist + self.b.unsqueeze(0) + self.gamma)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── ConvE ───────────────────────────────────────────────────────────────────
class ConvE(BaseKGE):
    """Dettmers et al., AAAI 2018. 2D convolution on reshaped (h,r)."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R = config['entity_cnt'], config['relation_cnt']
        d  = p.get('emb_dim', 200)
        sh = p.get('reshape', [10, 20])
        ks = p.get('conv_kernel_size', [3, 3])
        C  = p.get('conv_out_channels', 32)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.sh = sh
        self.E = nn.Embedding(N, d).to(self.dev); self.R = nn.Embedding(R, d).to(self.dev)
        self.inp_drop  = nn.Dropout(p.get('input_dropout', 0.2)).to(self.dev)
        self.feat_drop = nn.Dropout2d(p.get('feature_map_dropout', 0.2)).to(self.dev)
        self.hid_drop  = nn.Dropout(p.get('hidden_dropout', 0.3)).to(self.dev)
        self.bn0 = nn.BatchNorm2d(1).to(self.dev)
        self.bn1 = nn.BatchNorm2d(C).to(self.dev)
        self.bn2 = nn.BatchNorm1d(d).to(self.dev)
        fc_in = C * (2*sh[0]-ks[0]+1) * (sh[1]-ks[1]+1)
        self.conv = nn.Conv2d(1, C, ks).to(self.dev)
        self.fc   = nn.Linear(fc_in, d).to(self.dev)
        self.b    = Parameter(torch.zeros(N)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)

    def forward(self, bh, br, bt=None, inverse=False):
        h = self.E(bh); r = self.R(br)
        if inverse: r = -r
        stacked = torch.cat([h.view(-1,1,*self.sh), r.view(-1,1,*self.sh)], 2)
        stacked = self.inp_drop(self.bn0(stacked))
        x = self.feat_drop(F.relu(self.bn1(self.conv(stacked))))
        x = F.relu(self.bn2(self.hid_drop(self.fc(x.view(x.size(0),-1)))))
        scores = torch.sigmoid(torch.mm(x, self.E.weight.T) + self.b)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── ConvKB ──────────────────────────────────────────────────────────────────
class ConvKB(BaseKGE):
    """Nguyen et al., NAACL 2018. 1D conv over triple representation."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R, d = config['entity_cnt'], config['relation_cnt'], p.get('emb_dim',200)
        C = p.get('conv_out_channels', 50)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.E = nn.Embedding(N, d).to(self.dev); self.R = nn.Embedding(R, d).to(self.dev)
        self.conv = nn.Conv1d(3, C, kernel_size=1).to(self.dev)
        self.fc   = nn.Linear(d*C, d).to(self.dev)
        self.b    = nn.Parameter(torch.zeros(N)).to(self.dev)
        self.drop = nn.Dropout(p.get('input_dropout', 0.5)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)

    def forward(self, bh, br, bt=None, inverse=False):
        h = self.E(bh); r = self.R(br)
        if inverse: r = -r
        B = bh.size(0)
        triple = self.drop(torch.stack([h, r, torch.zeros_like(h)], dim=1))  # (B,3,d)
        x = F.relu(self.conv(triple)).view(B,-1)
        ctx = self.drop(self.fc(x))
        scores = torch.sigmoid(torch.mm(ctx, self.E.weight.T) + self.b)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── InteractE ───────────────────────────────────────────────────────────────
class InteractE(BaseKGE):
    """Vashishth et al., AAAI 2020. Feature permutation + circular convolution."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R = config['entity_cnt'], config['relation_cnt']
        d  = p.get('emb_dim', 200)
        sh = p.get('reshape', [10, 20])
        ks = p.get('conv_kernel_size', [9, 11])
        C  = p.get('conv_out_channels', 96)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.sh = sh; self.ph = ks[0]//2; self.pw = ks[1]//2
        self.E = nn.Embedding(N, d).to(self.dev); self.R = nn.Embedding(R, d).to(self.dev)
        self.inp_drop  = nn.Dropout(p.get('input_dropout', 0.2)).to(self.dev)
        self.feat_drop = nn.Dropout2d(p.get('feature_map_dropout', 0.2)).to(self.dev)
        self.hid_drop  = nn.Dropout(p.get('hidden_dropout', 0.3)).to(self.dev)
        self.bn0 = nn.BatchNorm2d(1).to(self.dev)
        self.bn1 = nn.BatchNorm2d(C).to(self.dev)
        self.bn2 = nn.BatchNorm1d(d).to(self.dev)
        fc_in = C * 2*sh[0] * sh[1]
        self.conv = nn.Conv2d(1, C, ks, padding=0).to(self.dev)
        self.fc   = nn.Linear(fc_in, d).to(self.dev)
        self.b    = Parameter(torch.zeros(N)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)

    def _circ_pad(self, x):
        x = torch.cat([x[...,-self.pw:], x, x[...,:self.pw]], dim=-1)
        x = torch.cat([x[...,-self.ph:,:], x, x[...,:self.ph,:]], dim=-2)
        return x

    def forward(self, bh, br, bt=None, inverse=False):
        h = self.E(bh); r = self.R(br)
        if inverse: r = -r
        # interleave h and r features
        hr = torch.stack([h, r], dim=2).reshape(bh.size(0), 2, -1)
        h2 = hr[:,0,:].view(-1,1,*self.sh); r2 = hr[:,1,:].view(-1,1,*self.sh)
        stacked = self.inp_drop(self.bn0(torch.cat([h2, r2], dim=2)))  # (B,1,2H,W)
        x = self.feat_drop(F.relu(self.bn1(self.conv(self._circ_pad(stacked)))))
        x = F.relu(self.bn2(self.hid_drop(self.fc(x.view(x.size(0),-1)))))
        scores = torch.sigmoid(torch.mm(x, self.E.weight.T) + self.b)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ─── KGCN ────────────────────────────────────────────────────────────────────
class KGCN(BaseKGE):
    """Wang et al., WWW 2019. Multi-hop relation-attentive neighborhood aggregation."""
    def __init__(self, config):
        super().__init__(config)
        p = config['model_hyper_params']
        N, R = config['entity_cnt'], config['relation_cnt']
        d  = p.get('emb_dim', 64)
        K  = p.get('neighbor_sample_size', 8)
        n_iter = p.get('n_iter', 2)
        self.dev = config['device']; self.N = N; self.ls = p.get('label_smoothing',0.1)
        self.n_iter = n_iter
        self.E  = nn.Embedding(N+1, d, padding_idx=N).to(self.dev)
        self.R  = nn.Embedding(R+1, d, padding_idx=R).to(self.dev)
        self.W  = nn.ModuleList([nn.Linear(d,d,bias=False).to(self.dev) for _ in range(n_iter)])
        self.b  = nn.Parameter(torch.zeros(N)).to(self.dev)
        self.drop = nn.Dropout(p.get('input_dropout',0.5)).to(self.dev)
        nn.init.xavier_uniform_(self.E.weight); nn.init.xavier_uniform_(self.R.weight)
        # build neighbor table
        self.nb_ent, self.nb_rel = self._build_neighbors(config.get('data',[]), N, R, K)

    def _build_neighbors(self, data, N, R, K):
        from collections import defaultdict
        adj = defaultdict(list)
        for h,t,r in data:
            adj[h].append((t,r)); adj[t].append((h,r))
        nb_e = torch.full((N+1,K), N, dtype=torch.long)
        nb_r = torch.full((N+1,K), R, dtype=torch.long)
        for e in range(N):
            ns = adj[e]
            if not ns: continue
            idx = torch.randint(len(ns),(K,))
            for k,i in enumerate(idx): nb_e[e,k]=ns[i][0]; nb_r[e,k]=ns[i][1]
        return nb_e.to(self.dev), nb_r.to(self.dev)

    def _agg(self, ents, r_emb):
        nbe = self.nb_ent[ents]; nbr = self.nb_rel[ents]
        ne = self.E(nbe); nr = self.R(nbr)
        attn = F.softmax((nr * r_emb.unsqueeze(1)).sum(-1), dim=-1)
        return (attn.unsqueeze(-1)*ne).sum(1)

    def forward(self, bh, br, bt=None, inverse=False):
        r_e = self.R(br)
        if inverse: r_e = -r_e
        x = self.E(bh)
        for i in range(self.n_iter):
            x = F.relu(self.W[i](x + self._agg(bh, r_e)))
            x = self.drop(x)
        scores = torch.sigmoid(torch.mm(x*r_e, self.E.weight[:self.N].T) + self.b)
        return (self._bce_loss(scores,bt,self.N,self.dev,self.ls) if bt is not None else None), scores


# ═══════════════════════════════════════════════════════════════════════════════
#  模型注册表 & 默认超参数
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    'TransE':    TransE,
    'TransH':    TransH,
    'TransR':    TransR,
    'TransD':    TransD,
    'DistMult':  DistMult,
    'ComplEx':   ComplEx,
    'RotatE':    RotatE,
    'ConvE':     ConvE,
    'ConvKB':    ConvKB,
    'InteractE': InteractE,
    'KGCN':      KGCN,
}

HYPER = {
    'TransE':    dict(emb_dim=200, label_smoothing=0.1),
    'TransH':    dict(emb_dim=200, label_smoothing=0.1),
    'TransR':    dict(emb_dim=200, rel_dim=200, label_smoothing=0.1),
    'TransD':    dict(emb_dim=200, rel_dim=200, label_smoothing=0.1),
    'DistMult':  dict(emb_dim=200, input_dropout=0.2, label_smoothing=0.1),
    'ComplEx':   dict(emb_dim=200, input_dropout=0.2, label_smoothing=0.1),
    'RotatE':    dict(emb_dim=500, gamma=12.0, label_smoothing=0.1),
    'ConvE':     dict(emb_dim=200, reshape=[10,20], conv_out_channels=32,
                      conv_kernel_size=[3,3], input_dropout=0.2,
                      feature_map_dropout=0.2, hidden_dropout=0.3, label_smoothing=0.1),
    'ConvKB':    dict(emb_dim=200, conv_out_channels=50,
                      input_dropout=0.5, label_smoothing=0.1),
    'InteractE': dict(emb_dim=200, reshape=[10,20], conv_out_channels=96,
                      conv_kernel_size=[9,11], input_dropout=0.2,
                      feature_map_dropout=0.2, hidden_dropout=0.3, label_smoothing=0.1),
    'KGCN':      dict(emb_dim=64, n_iter=2, neighbor_sample_size=8,
                      input_dropout=0.5, label_smoothing=0.1),
}

LR = {'TransE':5e-4,'TransH':5e-4,'TransR':5e-4,'TransD':5e-4,
      'DistMult':1e-3,'ComplEx':1e-3,'RotatE':5e-5,
      'ConvE':3e-3,'ConvKB':1e-4,'InteractE':3e-3,'KGCN':1e-3}

ALL_MODELS   = list(MODEL_REGISTRY.keys())
ALL_DATASETS = ['FB15K237', 'WN18RR', 'YAGO3-10']


# ═══════════════════════════════════════════════════════════════════════════════
#  评估 & 结果保存
# ═══════════════════════════════════════════════════════════════════════════════
def metrics(results):
    hits=np.array(results[0]); rr=np.array(results[4])
    return dict(MRR=(1/rr).mean(), H1=hits[0].mean(),
                H3=hits[2].mean(), H10=hits[9].mean(), MR=rr.mean())

def save_result(name, dataset, split, m, out_dir='baseline_results'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{dataset}.txt')
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path,'a',encoding='utf-8') as f:
        f.write(f'\n## {name} | {dataset} | {split}  — {ts}\n')
        f.write(f"- MRR    : {m['MRR']:.4f}\n- Hits@1 : {m['H1']:.4f}\n")
        f.write(f"- Hits@3 : {m['H3']:.4f}\n- Hits@10: {m['H10']:.4f}\n- MR: {m['MR']:.1f}\n")
    print(f"[{name}/{dataset}/{split}]  "
          f"MRR={m['MRR']:.4f}  H@1={m['H1']:.4f}  H@3={m['H3']:.4f}  "
          f"H@10={m['H10']:.4f}  MR={m['MR']:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  单次实验
# ═══════════════════════════════════════════════════════════════════════════════
def run_experiment(model_name, dataset_name, epochs, batch_size, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = Dataset(dataset_name)
    config = dict(
        device=device,
        entity_cnt=len(ds.data['entity']),
        relation_cnt=len(ds.data['relation']),
        data=ds.data['train'],
        model_hyper_params=HYPER[model_name],
    )

    model = MODEL_REGISTRY[model_name].init_model(config).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR[model_name], weight_decay=1e-5)

    tr = DataLoader(ds.data['train'], batch_size, shuffle=True,  drop_last=False)
    va = DataLoader(ds.data['valid'], batch_size, shuffle=False, drop_last=False)
    te = DataLoader(ds.data['test'],  batch_size, shuffle=False, drop_last=False)

    best_mrr = 0.0
    for ep in range(1, epochs+1):
        t0 = time.time()
        losses = _train_epoch(tr, model, opt, device)
        print(f'[{model_name}/{dataset_name}] ep {ep}/{epochs}  '
              f'loss={np.mean(losses):.4f}  ({time.time()-t0:.1f}s)')
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                r = _eval_tail(va, model, device, ds.data)
            m = metrics(r)
            print(f'  [Valid] MRR={m["MRR"]:.4f}  H@1={m["H1"]:.4f}  '
                  f'H@3={m["H3"]:.4f}  H@10={m["H10"]:.4f}')
            best_mrr = max(best_mrr, m['MRR'])
            model.train()

    model.eval()
    with torch.no_grad():
        r = _eval_tail(te, model, device, ds.data)
    m = metrics(r)
    save_result(model_name, dataset_name, 'test', m)

    os.makedirs('./output/baselines', exist_ok=True)
    torch.save(model.state_dict(),
               f'./output/baselines/{model_name}_{dataset_name}.pt')
    return m


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--dataset',    default='FB15K237',
                    help='Dataset or "all". Options: '+', '.join(ALL_DATASETS))
    pa.add_argument('--model',      default='TransE',
                    help='Model or "all". Options: '+', '.join(ALL_MODELS))
    pa.add_argument('--epochs',     type=int, default=200)
    pa.add_argument('--batch_size', type=int, default=128)
    pa.add_argument('--seed',       type=int, default=42)
    args = pa.parse_args()

    datasets = ALL_DATASETS if args.dataset=='all' else [args.dataset]
    models   = ALL_MODELS   if args.model  =='all' else [args.model]

    all_res = {d:{} for d in datasets}
    for ds in datasets:
        for mn in models:
            logging.info(f'{"="*60}\nRunning {mn} on {ds}\n{"="*60}')
            try:
                all_res[ds][mn] = run_experiment(
                    mn, ds, args.epochs, args.batch_size, args.seed)
            except Exception as e:
                logging.error(f'[{mn}/{ds}] FAILED: {e}')
                import traceback; traceback.print_exc()
                all_res[ds][mn] = None

    # 打印汇总表
    print('\n'+'='*70+'\nRESULTS SUMMARY\n'+'='*70)
    for ds, mods in all_res.items():
        print(f'\n### {ds}\n')
        print(f'{"Model":<12} {"MRR":>7} {"H@1":>7} {"H@3":>7} {"H@10":>7} {"MR":>8}')
        print('-'*50)
        for mn, m in mods.items():
            if m: print(f'{mn:<12} {m["MRR"]:>7.4f} {m["H1"]:>7.4f} '
                        f'{m["H3"]:>7.4f} {m["H10"]:>7.4f} {m["MR"]:>8.1f}')

    os.makedirs('baseline_results', exist_ok=True)
    with open('baseline_results/summary.json','w') as f:
        json.dump(all_res, f, indent=2)
    print('\nSaved → baseline_results/summary.json')

if __name__ == '__main__':
    main()
