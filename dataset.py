import json
import random
import gzip
import requests

def product_to_chunk(product):
    title = product["title"]
    features = ", ".join(product.get("features", []))
    description = "; ".join(product.get("description", []))
    categories = ", ".join(product.get("categories", []))
    details = product.get("details", {})
    details_str = ", ".join([f"{k}:{v}" for k, v in details.items()])

    text = (
        f"Product Title: {title}; "
        f"Categories: {categories}; "
        f"Features: {features}; "
        f"Description: {description}; "
        f"Specs: {details_str}; "
        f"Price: {product.get('price')}; "
        f"Average rating: {product.get('average_rating')}; "
        f"Review count: {product.get('rating_number')}; "
        f"Store: {product.get('store')}"
    )
    return {
        "id": product["parent_asin"],
        "text": text,
        "meta": {"parent_asin": product["parent_asin"]}
    }

# Amazon 2023 Clothing 元数据链接
url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz"

candidate_pool = []
target_total = 10000

print("Start streaming product metadata ...")
with requests.get(url, stream=True, timeout=600) as r:
    r.raise_for_status()
    with gzip.GzipFile(fileobj=r.raw) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prod = json.loads(line)
            # 基础过滤，剔除无效商品
            if "parent_asin" not in prod or "title" not in prod or len(prod["title"].strip()) == 0:
                continue
            candidate_pool.append(prod)
            # 读到2倍目标数量就停止读取流，节约网络和内存
            if len(candidate_pool) >= target_total * 2:
                break

# 随机采样10000条，seed固定保证复现
random.seed(42)
sample_products = random.sample(candidate_pool, k=target_total)
kb_chunks = [product_to_chunk(p) for p in sample_products]

# 输出知识库文件
with open("kb.json", "w", encoding="utf-8") as f:
    json.dump(kb_chunks, f, indent=2, ensure_ascii=False)

with open("kb_pyserini.jsonl", "w", encoding="utf-8") as f:
    for chunk in kb_chunks:
        out = json.dumps({"id": chunk["id"], "contents": chunk["text"]}, ensure_ascii=False)
        f.write(out + "\n")

# 统计单词数，用于作业报告
def count_words(s):
    return len(s.split())
total_words = sum(count_words(c["text"]) for c in kb_chunks)
print(f"✅ Corpus built. Documents: {len(kb_chunks)}, total words: {total_words}")
