import torch
import logging
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

def save_array_with_numpy(array, filename):
    """
    使用 numpy 保存数组到文件
    :param array: 要保存的数组
    :param filename: 保存的文件名
    """
    np.save(filename, array)
    print(f"数组已成功保存到 {filename}.npy")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_for_tail(eval_data, model, device, data, descending, raoit=0):
    hits = []
    hits_left = []
    hits_right = []
    ranks = []
    ranks_left = []
    ranks_right = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
    ent_rel_multi_h = data['entity_relation']['as_head']
    for _ in range(10):  # need at most Hits@10
        hits.append([])
        hits_left.append([])
        hits_right.append([])

    de_rank0 = []
    de_rank10 = []

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred = model(eval_h, eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity
        _, pred1 = model(eval_t, eval_r, inverse=True)

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]
            filter_h = ent_rel_multi_h[eval_t[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_t[i].item()].item()
            pred_value1 = pred1[i][eval_h[i].item()].item()
            pred[i][filter_t] = 0.0
            pred1[i][filter_h] = 0.0
            pred[i][eval_t[i].item()] = pred_value
            pred1[i][eval_h[i].item()] = pred_value1

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        _, index1 = torch.sort(pred1, 1, descending=True)
        index = index.cpu().numpy()  # index: (batch_size)
        index1 = index1.cpu().numpy()

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_t[i].item())[0][0]
            rank1 = np.where(index1[i] == eval_h[i].item())[0][0]

            if rank == 0:
                de_rank0.append((eval_h[i].cpu(), eval_t[i].cpu(), eval_r[i].cpu()))
            if rank > 9:
                de_rank10.append((eval_h[i].cpu(), eval_t[i].cpu(), eval_r[i].cpu()))


            # rank+1, since the rank starts with 1 not 0
            ranks_left.append(rank1 + 1)
            ranks_right.append(rank + 1)
            ranks.append(rank1 + 1)
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

            for hits_level in range(10):
                if rank1 <= hits_level:
                    hits_left[hits_level].append(1.0)
                else:
                    hits_left[hits_level].append(0.0)

    save_array_with_numpy(de_rank0, "rank0.npy")
    save_array_with_numpy(de_rank10, "rank10.npy")

    return hits, hits_left, ranks, ranks_left, ranks_right

def output_eval_tail(
    results: list, 
    data_name: str, 
    model: str = "",
    data: str = "",
):
    file_path = os.path.join("result", f"{data}.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    mode = "a"
    


    hits = np.array(results[0])
    # hits_left = np.array(results[1])
    # ranks = np.array(results[2])
    # ranks_left = np.array(results[3])
    ranks_right = np.array(results[4])
    # r_ranks = 1.0 / ranks  # compute reciprocal rank
    # r_ranks_left = 1.0 / ranks_left
    r_ranks_right = 1.0 / ranks_right

    with open(file_path, mode, encoding="utf-8") as f:
        if mode == "a":     # 追加时补一个空行，保持排版
            f.write("\n")
        f.write(f"## {data_name}  —  {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"- **Hits@10**: {hits[9].mean():.4f}\n")
        f.write(f"- **Hits@3** : {hits[2].mean():.4f}\n")
        f.write(f"- **Hits@1** : {hits[0].mean():.4f}\n")
        f.write(f"- **MR**     : {ranks_right.mean():.4f}\n")
        f.write(f"- **MRR**    : {r_ranks_right.mean():.4f}\n\n")

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))

    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks_right.mean(), r_ranks_right.mean()))