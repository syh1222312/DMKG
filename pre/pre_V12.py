import random
import os
from typing import Dict
import ijson

json_path = '../INITDATASET/dblp-v12-clean.json'  # ←←← 改成你的 JSON 文件路径
output_dir = '../data/DBLP_V12'  # 输出目录
random_seed = 42
# ===============================================================

os.makedirs(output_dir, exist_ok=True)
random.seed(random_seed)

from gensim.parsing.preprocessing import STOPWORDS

STOPWORDS = set(STOPWORDS)               # convert once (frozenset → set)

# Add domain-specific noise words (very common pattern)
STOPWORDS.update({
    'also', 'get', 'got', 'make', 'go', 'going', 'one', 'two', 'way',
    'really', 'just', 'even', 'back', 'well', 'still', 'much', 'more'
    'if'
})

for w in ['no', 'not', 'never', 'cannot', 'can', 'could', 'would', 'should']:
    STOPWORDS.discard(w)
def normalize_entity_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    words = name.split()
    filtered = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    if not filtered:  # 如果全被过滤掉，保留原始（避免完全丢失）
        return name.replace(" ", "/")
    return "/".join(filtered)

# ------------------- Pass 1: 流式收集实体 -------------------
print("Pass 1 开始：收集所有实体（venue/fos 会规范化 + 去停用词）...")
entity2id: Dict[str, int] = {}
next_eid = 0

def add_entity(ent, normalize: bool = False):
    global next_eid
    if not ent:
        return
    key = str(ent)  # 统一转为 str，确保 key 是字符串
    if normalize:
        key = normalize_entity_name(key)
    if not key:
        return
    if key not in entity2id:
        entity2id[key] = next_eid
        next_eid += 1

line_count = 0
with open(json_path, 'r', encoding='utf-8') as f:
    for data in ijson.items(f, 'item'):  # 流式迭代数组中的每个对象
        line_count += 1
        if line_count % 100_000 == 0:
            print(f" 已处理 {line_count:,} 行 | 当前实体数: {len(entity2id):,} | 内存使用: {os.popen('ps -o rss= -p ' + str(os.getpid())).read().strip()} KB")

        # paper id （不需规范化）
        p_id = data.get('id')
        if p_id is not None:
            add_entity(p_id, normalize=False)

        # 作者 ID （不需规范化）
        for author in data.get('authors', []):
            a_id = author.get('id')
            if a_id is not None and str(a_id).strip():
                add_entity(a_id, normalize=False)

        # 引用论文（paper id，不需规范化）
        for ref in data.get('references', []):
            if ref is not None:
                add_entity(ref, normalize=False)

        # venue → 需要规范化，使用 raw
        venue = data.get('venue')
        if isinstance(venue, dict):
            v = venue.get('raw')
            if v and isinstance(v, str) and v.strip():
                add_entity(v, normalize=True)

        # fos → 需要规范化，使用 name 作为 keyword 等价
        for fos_item in data.get('fos', []):
            if isinstance(fos_item, dict):
                k = fos_item.get('name')
                if k and isinstance(k, str) and k.strip():
                    add_entity(k, normalize=True)

print(f"Pass 1 完成！共处理 {line_count:,} 行")
print(f"实体总数: {len(entity2id):,}（venue/fos 已空格转/ 并去停用词）")

# ------------------- 关系定义（含 coworker） -------------------
relations = ['author_of', 'cites', 'published_in', 'has_keyword', 'coworker']  # has_keyword 用 fos name
relation2id = {rel: idx for idx, rel in enumerate(relations)}

# ------------------- 输出 entity2id.txt -------------------
with open(os.path.join(output_dir, 'entity2id.txt'), 'w', encoding='utf-8') as f:
    f.write(f"{len(entity2id)}\n")
    for ent, eid in sorted(entity2id.items(), key=lambda x: x[1]):  # 按 id 排序，可选
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

with open(json_path, 'r', encoding='utf-8') as f:
    for data in ijson.items(f, 'item'):  # 流式迭代数组中的每个对象
        p_id = data.get('id')
        if p_id is None or str(p_id) not in entity2id:
            continue
        p_eid = entity2id[str(p_id)]

        # 1. author_of : 作者 → 论文
        for author in data.get('authors', []):
            a_id = author.get('id')
            if a_id is not None and str(a_id).strip() and str(a_id) in entity2id:
                bucket = random.randint(0, 9)
                h = entity2id[str(a_id)]  # 头：作者
                t = p_eid  # 尾：论文
                r = relation2id['author_of']
                bucket_files[bucket].write(f"{h} {t} {r}\n")
                counts[bucket] += 1

        # 2. cites : 论文 → 引用论文
        for ref in data.get('references', []):
            if ref is not None and str(ref) in entity2id:
                bucket = random.randint(0, 9)
                h = p_eid  # 头：当前论文
                t = entity2id[str(ref)]  # 尾：被引论文
                r = relation2id['cites']
                bucket_files[bucket].write(f"{h} {t} {r}\n")
                counts[bucket] += 1

        # 3. published_in : 论文 → 会议/期刊，使用 raw
        venue = data.get('venue')
        if isinstance(venue, dict):
            v = venue.get('raw')
            if v and isinstance(v, str) and v.strip():
                v_key = normalize_entity_name(v)
                if v_key in entity2id:
                    bucket = random.randint(0, 9)
                    h = p_eid  # 头：论文
                    t = entity2id[v_key]  # 尾：venue
                    r = relation2id['published_in']
                    bucket_files[bucket].write(f"{h} {t} {r}\n")
                    counts[bucket] += 1

        # 4. has_keyword : 论文 → fos name 作为关键词
        for fos_item in data.get('fos', []):
            if isinstance(fos_item, dict):
                k = fos_item.get('name')
                if k and isinstance(k, str) and k.strip():
                    k_key = normalize_entity_name(k)
                    if k_key in entity2id:
                        bucket = random.randint(0, 9)
                        h = p_eid  # 头：论文
                        t = entity2id[k_key]  # 尾：关键词
                        r = relation2id['has_keyword']
                        bucket_files[bucket].write(f"{h} {t} {r}\n")
                        counts[bucket] += 1

        # 5. coworker : 第一作者 → 其他合作作者（单向）
        author_ids = []
        for author in data.get('authors', []):
            a_id = author.get('id')
            if a_id is not None and str(a_id).strip() and str(a_id) in entity2id:
                author_ids.append(str(a_id))  # 转为 str
        author_ids = list(dict.fromkeys(author_ids))  # 去重，保留顺序
        if len(author_ids) >= 2:
            r = relation2id['coworker']
            first_author_id = author_ids[0]
            h = entity2id[first_author_id]  # 头：第一作者
            for j in range(1, len(author_ids)):
                other_author_id = author_ids[j]
                t = entity2id[other_author_id]  # 尾：其他作者
                bucket = random.randint(0, 9)
                bucket_files[bucket].write(f"{h} {t} {r}\n")
                counts[bucket] += 1

for f in bucket_files:
    f.close()

total_triples = sum(counts)
print(f"Pass 2 完成！总三元组数量: {total_triples:,}（coworker 已用 ID 连接）")

# ------------------- 严格 8:1:1 划分（使用桶合并） -------------------
split_configs = {
    'train2id.txt': list(range(8)),  # 80%
    'valid2id.txt': [8],  # 10%
    'test2id.txt': [9]  # 10%
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
    print(f" 已生成 {filename} → {split_total:,} 条")

# ------------------- 删除临时桶文件 -------------------
for p in bucket_paths:
    try:
        os.remove(p)
    except:
        pass
print("临时桶文件已删除")