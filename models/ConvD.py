import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from math import sqrt
from tqdm import tqdm
from sklearn.cluster import KMeans
from .BaseModel import BaseModel

class ConvD(BaseModel):
    def __init__(self, config):
        super(ConvD, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')
        self.conv_out_channels = kwargs.get('conv_out_channels')
        self.reshape = kwargs.get('reshape')
        self.kernel_size = kwargs.get('conv_kernel_size')
        self.stride = kwargs.get('stride')
        self.emb_dim = {
            'entity': kwargs.get('emb_dim'),
            'relation': self.conv_out_channels * self.kernel_size[0] * self.kernel_size[1]
        }
        self.reshape[0] = int(self.emb_dim['entity'] / self.reshape[1])
        assert self.emb_dim['entity'] == self.reshape[0] * self.reshape[1]
        self.E = torch.nn.Embedding(self.entity_cnt, self.emb_dim['entity']).to(self.device)
        self.R = torch.nn.Embedding(self.relation_cnt, self.emb_dim['relation']).to(self.device)
        self.q_size = kwargs.get('q_size')
        self.q_size[0] = self.emb_dim['entity']
        self.k_size = kwargs.get('k_size')
        self.k_size[0] = self.emb_dim['relation']
        self.v_size = kwargs.get('v_size')
        self.v_size[0] = self.emb_dim['relation']
        self.v_size[1] = self.conv_out_channels
        self.Q = nn.Parameter(torch.randn(self.q_size)).to(self.device)
        self.K = nn.Parameter(torch.randn(self.k_size)).to(self.device)
        self.V = nn.Parameter(torch.randn(self.v_size)).to(self.device)
        self.a = kwargs.get('a')
        self.mp_weight = kwargs.get('b', 0.1)  # 改名mp_weight

        # ==================== 新增：可学习原型 + projector（方案一，最终推荐版） ====================
        self.num_clusters = kwargs.get('num_clusters', 8)          # 推荐值：8
        self.tau = kwargs.get('tau', 0.08)                         # 推荐值：0.08
        self.mask_prob = kwargs.get('mask_prob', 0.10)             # 推荐值：0.10
        self.gcl_weight = kwargs.get('gcl_weight', 0.15)           # 推荐值：0.15（大幅降低）
        self.cluster_weight = kwargs.get('cluster_weight', 0.008)  # 推荐值：0.008（大幅降低）
        self.cluster_alpha = kwargs.get('cluster_alpha', 1.0)

        # 可学习原型（全局共享，取代 KMeans）
        self.cluster_prototypes = nn.Parameter(
            torch.randn(self.num_clusters, self.emb_dim['entity'])
        ).to(self.device)
        torch.nn.init.xavier_normal_(self.cluster_prototypes)

        # GCL projector（两层 MLP，防止 collapse）
        self.projector = nn.Sequential(
            nn.Linear(self.emb_dim['entity'], self.emb_dim['entity']),
            nn.ReLU(),
            nn.Linear(self.emb_dim['entity'], self.emb_dim['entity']),
        ).to(self.device)
        # ================================================================================

        self.P = torch.zeros(config.get('entity_cnt'), config.get('relation_cnt')).to(self.device)
        self.MP = torch.zeros(self.entity_cnt, self.relation_cnt).to(self.device)
        self.attention(config.get('data'))
        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout')).to(self.device)
        self.feature_map_drop = torch.nn.Dropout2d(kwargs.get('feature_map_dropout')).to(self.device)
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout')).to(self.device)
        self.bn0 = torch.nn.BatchNorm2d(1).to(self.device)  # batch normalization over a 4D input
        self.bn1 = torch.nn.BatchNorm2d(self.conv_out_channels).to(self.device)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim['entity']).to(self.device)
        self.bn3 = torch.nn.BatchNorm1d(self.emb_dim['relation']).to(self.device)
        self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.filtered = [(self.reshape[0] - self.kernel_size[0]) // self.stride + 1, (self.reshape[1] - self.kernel_size[1]) // self.stride + 1]
        fc_length = self.filtered[0] * self.filtered[1]
        self.fc = torch.nn.Linear(fc_length, self.emb_dim['entity']).to(self.device)
        self.loss = ConvDLoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    def attention(self, data):
        max_size = self.config.get('model_hyper_params').get('memory_size', 10000)  # 背包大小
        M = []
        freq = {}
        for d in tqdm(range(len(data))):
            h = data[d][0]
            t = data[d][1]
            r = data[d][2]
            has_hr = any(m[0] == h and m[1] == r for m in M)
            if has_hr:
                has_o = any(m[2] == t for m in M)
                if has_o:
                    if (h, r, t) not in M:
                        M.append((h, r, t))
                        freq[t] = freq.get(t, 0) + 1
                else:
                    if len(M) >= max_size:
                        if freq:
                            min_o = min(freq, key=freq.get)
                            for i, m in enumerate(M):
                                if m[2] == min_o:
                                    del M[i]
                                    freq[min_o] -= 1
                                    if freq[min_o] == 0:
                                        del freq[min_o]
                                    break
                    M.append((h, r, t))
                    freq[t] = freq.get(t, 0) + 1
            else:
                if len(M) >= max_size:
                    if freq:
                        min_o = min(freq, key=freq.get)
                        for i, m in enumerate(M):
                            if m[2] == min_o:
                                del M[i]
                                freq[min_o] -= 1
                                if freq[min_o] == 0:
                                    del freq[min_o]
                                break
                M.append((h, r, t))
                freq[t] = freq.get(t, 0) + 1

        for d in tqdm(range(len(data))):
            h = data[d][0]
            t = data[d][1]
            r = data[d][2]
            if (h, r, t) in M:
                self.MP[h][r] += 1
            else:
                self.P[h][r] += 1

        for i in tqdm(range(self.P.size(0))):
            for j in range(self.P.size(1)):
                self.P[i][j] = torch.log(self.P[i][j] + 1)
                self.MP[i][j] = torch.log(self.MP[i][j] + 1)

    def forward(self, batch_h, batch_r, batch_t=None, inverse=False):
        batch_size = batch_h.size(0)
        E = self.E(torch.tensor(batch_h))
        R = self.R(torch.tensor(batch_r))
        Q = torch.mm(E, self.Q)
        K = torch.mm(R, self.K)
        V = torch.mm(R, self.V)
        res = (torch.mm(Q, K.T) / sqrt(self.q_size[1]) + self.a * torch.tensor(self.P[batch_h][:, batch_r]) + self.mp_weight * torch.tensor(self.MP[batch_h][:, batch_r]))
        res = torch.softmax(res, dim=1)
        atten = torch.mm(res, V)
        e1 = self.E(batch_h).view(-1, 1, *self.reshape)
        e1 = self.bn0(e1).view(1, -1, *self.reshape)
        e1 = self.input_drop(e1)
        r = self.R(batch_r)
        if inverse==True:
            r = -r
        r = self.bn3(r)
        r = self.input_drop(r)
        r = r.view(-1, 1, *self.kernel_size)
        x = F.conv2d(e1, r, groups=batch_size)
        x = x.view(batch_size, self.conv_out_channels, *self.filtered)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = atten.view(batch_size, self.conv_out_channels, 1, 1) * x
        x = x.sum(dim=1)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        y = torch.sigmoid(x)
        return self.loss(y, batch_t), y


    def compute_gcl_and_cluster_loss(self, batch_h, batch_t, epoch=None):
        """
        epoch 参数用于 warmup（前 5 个 epoch 不计算辅助损失）
        如果你的训练循环没有传 epoch，可以直接删掉 epoch 相关代码。
        """
        # 1. warmup：前 5 个 epoch 只算主损失（避免早期梯度冲突）
        if epoch is not None and epoch < 5:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        # 2. 提取 batch 内 unique 实体
        unique_nodes = torch.unique(torch.cat((batch_h, batch_t)))
        n_unique = len(unique_nodes)
        if n_unique < 2:
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero

        emb_orig = self.E(unique_nodes)  # (N, D) 原始 embedding

        # ---------- 1. 可学习原型聚类损失 ----------
        diff = emb_orig.unsqueeze(1) - self.cluster_prototypes.unsqueeze(0)  # (N, K, D)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # (N, K)
        p = (1.0 + dist_sq / self.cluster_alpha) ** (-0.5 * (self.cluster_alpha + 1))
        p = p / (p.sum(dim=1, keepdim=True) + 1e-9)

        f = p.sum(dim=0)  # (K,)
        q = p ** 2 / (f.unsqueeze(0) + 1e-9)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-9)

        cluster_loss = torch.sum(q * torch.log(q / (p + 1e-9) + 1e-9)) / n_unique

        # ---------- 2. 标准 InfoNCE（两视图 + 温和 dropout） ----------
        emb_view2 = F.dropout(emb_orig, p=self.mask_prob, training=True)

        z1 = F.normalize(self.projector(emb_orig), dim=-1)
        z2 = F.normalize(self.projector(emb_view2), dim=-1)

        # view1 → view2
        sim12 = torch.matmul(z1, z2.T) / self.tau
        labels = torch.arange(n_unique, device=self.device)
        gcl12 = F.cross_entropy(sim12, labels)

        # view2 → view1（对称）
        sim21 = torch.matmul(z2, z1.T) / self.tau
        gcl21 = F.cross_entropy(sim21, labels)

        gcl_loss = (gcl12 + gcl21) / 2

        return gcl_loss, cluster_loss

class ConvDLoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt):
        super().__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt

    def forward(self, batch_p, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            batch_e = torch.zeros(batch_size, self.entity_cnt).to(self.device).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss = self.loss(batch_p, batch_e) / batch_size
        return loss