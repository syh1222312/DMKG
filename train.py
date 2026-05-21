import torch
import logging 
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_without_label(data, model, optimizer,epoch, device):
    full_loss = []
    model.train()
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)

        optimizer.zero_grad()
        # 正向 + 逆向主任务损失（保持原逻辑）
        loss1, _ = model(t, r, h, True)
        loss1 = loss1.mean()
        loss2, _ = model(h, r, t)
        loss2 = loss2.mean()

        # ==================== GCL + 聚类损失（使用我们修改后的版本） ====================
        # 注意：必须传入 epoch 参数才能触发 warmup（前5个epoch不计算辅助损失）
        gcl_loss, cluster_loss = model.compute_gcl_and_cluster_loss(h, t, epoch=epoch)

        gcl_total = model.gcl_weight * gcl_loss + model.cluster_weight * cluster_loss

        # 总损失 = 主任务损失（正向+逆向） + GCL + 聚类损失
        total_loss = loss1 + loss2 + gcl_total
        # =============================================================================

        total_loss.backward()
        optimizer.step()

        full_loss.append(total_loss.item())  # 现在记录完整总损失（包含三部分）
    return full_loss