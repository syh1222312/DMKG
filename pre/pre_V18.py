import json
import random
import os
from typing import Dict

# ====================== 配置区（请修改这里） ======================
jsonl_path = '../INITDATASET/DBLP-Citation-network-V18.jsonl'  # ←←← 改成你的 15GB JSONL 文件路径
output_dir = '../data/DBLP_V18'  # 输出目录
random_seed = 42
# ===============================================================

os.makedirs(output_dir, exist_ok=True)
random.seed(random_seed)

# ------------------- 常用的英文停用词（可自行增减） -------------------
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'for', 'of', 'in', 'on', 'at', 'to', 'by',
    'with', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
    'this', 'that', 'these', 'those', 'it', 'its', 'their', 'our', 'my', 'your',
    'i', 'you', 'he', 'she', 'we', 'they', 'who', 'which', 'what', 'where',
    'when', 'how', 'why', 'all', 'any', 'some', 'no', 'not'
}


def normalize_entity_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""

    words = name.split()
    filtered = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]

    if not filtered:
        # 如果全被過濾掉，保留原始（避免完全丟失）
        return name.replace(" ", "/")

    return "/".join(filtered)


# ------------------- Pass 1: 流式收集实体 -------------------
print("Pass 1 开始：收集所有实体（venue/keyword 会规范化 + 去停用词）...")
entity2id: Dict[str, int] = {}
next_eid = 0


def add_entity(ent: str, normalize: bool = False):
    global next_eid
    if not ent:
        return

    key = ent

    # 只對 venue 和 keyword 做 空格→/ + 去停用詞 的處理
    if normalize:
        key = normalize_entity_name(ent)
        if not key:
            return

    if key not in entity2id:
        entity2id[key] = next_eid
        next_eid += 1


line_count = 0
filtered_count = 0
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

            # 新增过滤：作者数量 < 2 或 引用为空，则跳过
        authors = data.get('authors', [])
        references = data.get('references', [])
        if len(authors) < 2 or not references:
            filtered_count += 1
            continue

        line_count += 1
        if line_count % 100_000 == 0:
            print(f" 已处理 {line_count:,} 行 | 当前实体数: {len(entity2id):,}")

        # paper id （不需規範化）
        p_id = data.get('id')
        if p_id and isinstance(p_id, str):
            add_entity(p_id, normalize=False)

        # 作者 ID （不需規範化）
        for author in data.get('authors', []):
            a_id = author.get('id')
            if a_id and isinstance(a_id, str) and a_id.strip():
                add_entity(a_id, normalize=False)

        # 引用论文（paper id，不需規範化）
        for ref in data.get('references', []):
            if ref and isinstance(ref, str):
                add_entity(ref, normalize=False)

        # venue → 需要規範化
        venue = data.get('venue')
        if venue and isinstance(venue, str):
            v = venue.strip()
            if v:
                add_entity(v, normalize=True)

        # keywords → 需要規範化
        for kw in data.get('keywords', []):
            if kw and isinstance(kw, str):
                k = kw.strip()
                if k:
                    add_entity(k, normalize=True)

print(f"Pass 1 完成！共处理 {line_count:,} 行")
print(f"实体总数: {len(entity2id):,}（venue/keyword 已空格转/ 并去停用词）")

# ------------------- 关系定义（含 coworker） -------------------
relations = ['author_of', 'cites', 'published_in', 'has_keyword','coworker']
relation2id = {rel: idx for idx, rel in enumerate(relations)}

# ------------------- 输出 entity2id.txt -------------------
with open(os.path.join(output_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{len(entity2id)}\n")
    for ent, eid in sorted(entity2id.items(), key=lambda x: x[1]):  # 按 id 排序，可選
        f.write(f"{ent} {eid}\n")

# ------------------- 输出 relation2id.txt -------------------
with open(os.path.join(output_dir, 'relation2id.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{len(relations)}\n")
    for i, rel in enumerate(relations):
        f.write(f"{rel} {i}\n")


# ------------------- Pass 2: 流式生成三元组（coworker 用 ID 连接） -------------------
bucket_paths = [os.path.join(output_dir, f'temp_triples_bucket_{i}.txt') for i in range(10)]
bucket_files = [open(p, 'w', encoding='utf-8') for p in bucket_paths]
counts = [0] * 10

with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        p_id = data.get('id')
        if not p_id or p_id not in entity2id: continue
        p_eid = entity2id[p_id]

        # 1. author_of : 作者 → 论文
        for author in data.get('authors', []):
            a_id = author.get('id')
            if a_id and isinstance(a_id, str) and a_id != "" and a_id in entity2id:
                bucket = random.randint(0, 9)
                h = entity2id[a_id]           # 头：作者
                t = p_eid                     # 尾：论文
                r = relation2id['author_of']
                bucket_files[bucket].write(f"{h} {t} {r}\n")   # 修改为 h t r
                counts[bucket] += 1

        # 2. cites : 论文 → 引用论文
        for ref in data.get('references', []):
            if ref and ref in entity2id:
                bucket = random.randint(0, 9)
                h = p_eid                     # 头：当前论文
                t = entity2id[ref]            # 尾：被引论文
                r = relation2id['cites']
                bucket_files[bucket].write(f"{h} {t} {r}\n")   # 修改为 h t r
                counts[bucket] += 1

        # 3. published_in : 论文 → 会议/期刊
        venue = data.get('venue')
        if venue and isinstance(venue, str):
            v = venue.strip()
            if v and v in entity2id:
                bucket = random.randint(0, 9)
                h = p_eid                     # 头：论文
                t = entity2id[v]              # 尾：venue
                r = relation2id['published_in']
                bucket_files[bucket].write(f"{h} {t} {r}\n")   # 修改为 h t r
                counts[bucket] += 1

        # 4. has_keyword : 论文 → 关键词
        for kw in data.get('keywords', []):
            if kw and isinstance(kw, str):
                k = kw.strip()
                if k and k in entity2id:
                    bucket = random.randint(0, 9)
                    h = p_eid                     # 头：论文
                    t = entity2id[k]              # 尾：关键词
                    r = relation2id['has_keyword']
                    bucket_files[bucket].write(f"{h} {t} {r}\n")   # 修改为 h t r
                    counts[bucket] += 1

        # 5. coworker : 第一作者 → 其他合作作者（单向）
        author_ids = []
        for author in data.get('authors', []):
            a_id = author.get('id')
            if a_id and isinstance(a_id, str) and a_id.strip() and a_id in entity2id:
                author_ids.append(a_id)

        author_ids = list(dict.fromkeys(author_ids))  # 去重，保留顺序

        if len(author_ids) >= 2:
            r = relation2id['coworker']
            first_author_id = author_ids[0]
            h = entity2id[first_author_id]   # 头：第一作者

            for j in range(1, len(author_ids)):
                other_author_id = author_ids[j]
                t = entity2id[other_author_id]  # 尾：其他作者
                bucket = random.randint(0, 9)
                bucket_files[bucket].write(f"{h} {t} {r}\n")   # 修改为 h t r
                counts[bucket] += 1

for f in bucket_files:
    f.close()

total_triples = sum(counts)
print(f"Pass 2 完成！总三元组数量: {total_triples:,}（coworker 已用 ID 连接）")

# ------------------- 严格 8:1:1 划分（使用桶合并） -------------------
split_configs = {
    'train2id.txt': list(range(8)),   # 80%
    'valid2id.txt': [8],              # 10%
    'test2id.txt':  [9]               # 10%
}

for filename, bucket_list in split_configs.items():
    split_total = sum(counts[i] for i in bucket_list)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as out_f:
        out_f.write(f"{split_total}\n")
        for b in bucket_list:
            with open(bucket_paths[b], 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    out_f.write(line)
    print(f"  已生成 {filename} → {split_total:,} 条")

# ------------------- 校验 relation ID 分布（强烈推荐保留） -------------------
from collections import Counter
rel_counts = Counter()

for filename in ['train2id.txt', 'valid2id.txt', 'test2id.txt']:
    path = os.path.join(output_dir, filename)
    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过第一行总数
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    h, t, r = map(int, parts)   # 注意这里解析顺序也要改成 h t r
                    rel_counts[r] += 1
                except ValueError:
                    print(f"非法行 in {filename}: {line.strip()}")
            else:
                print(f"格式错误行 in {filename}: {line.strip()}")

print("所有 relation ID 出现次数：", dict(rel_counts))
print("出现的 relation ID 集合：", sorted(rel_counts.keys()))

# ------------------- 删除临时桶文件 -------------------
for p in bucket_paths:
    try:
        os.remove(p)
    except:
        pass
print("临时桶文件已删除")


