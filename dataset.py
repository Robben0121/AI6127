import os
import json
import random
import gzip
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Amazon 2023 元数据根地址（McAuley Lab）
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"

# 三个分类数据集
CATEGORIES = [
    "Clothing_Shoes_and_Jewelry",
    "Home_and_Kitchen",
    "Toys_and_Games",
]

PER_CATEGORY = 5000   # 每类最终采样条数
OVERSAMPLE = 2        # 每类先收集 2 倍候选，再随机采样，保证随机性
RANDOM_SEED = 42      # 固定随机种子，保证可复现


# ---------------- 字段校验工具 ----------------

def _is_nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""


def _nonempty_strings(lst):
    """从 list 中过滤出非空字符串条目（同时去重，保持顺序）。"""
    if not isinstance(lst, list):
        return []
    seen = set()
    out = []
    for x in lst:
        if isinstance(x, str) and x.strip():
            s = x.strip()
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _is_valid_price(v):
    """价格：必须是正数（支持带 $ 或逗号的字符串）。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        s = v.strip().replace("$", "").replace(",", "").strip()
        if not s:
            return False
        try:
            return float(s) > 0
        except ValueError:
            return False
    return False


def _is_positive_number(v):
    """数值字段（评分/评论数）：必须为大于 0 的数字。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        try:
            return float(v.strip()) > 0
        except ValueError:
            return False
    return False


def _details_nonempty(details):
    """details 字段：必须是非空 dict，且至少有一个非空值。"""
    if not isinstance(details, dict) or not details:
        return False
    for k, v in details.items():
        if k is None or (isinstance(k, str) and not k.strip()):
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        return True
    return False


# ---------------- 商品校验 ----------------

def product_valid(product):
    """校验商品：所有字段必须有数据、不能为空（用于过滤）。"""
    if not isinstance(product, dict):
        return False
    if not _is_nonempty_str(product.get("parent_asin")):
        return False
    if not _is_nonempty_str(product.get("title")):
        return False
    if not _nonempty_strings(product.get("features")):
        return False
    if not _nonempty_strings(product.get("description")):
        return False
    if not _nonempty_strings(product.get("categories")):
        return False
    if not _details_nonempty(product.get("details")):
        return False
    if not _is_valid_price(product.get("price")):
        return False
    if not _is_positive_number(product.get("average_rating")):
        return False
    if not _is_positive_number(product.get("rating_number")):
        return False
    if not _is_nonempty_str(product.get("store")):
        return False
    return True


def product_to_chunk(product):
    title = product["title"].strip()
    features = ", ".join(_nonempty_strings(product.get("features", [])))
    description = "; ".join(_nonempty_strings(product.get("description", [])))
    categories = ", ".join(_nonempty_strings(product.get("categories", [])))
    details = product.get("details", {})
    details_str = ", ".join(
        f"{k}:{v}" for k, v in details.items()
        if k is not None and not (isinstance(k, str) and not k.strip())
        and v is not None and not (isinstance(v, str) and not v.strip())
    )

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


# ---------------- 流式读取单个分类 ----------------

def load_category(category, target, seen_asins, rng):
    url = BASE_URL + f"meta_{category}.jsonl.gz"
    pool = []
    scanned = 0
    print(f"[{category}] streaming {url} ...")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                scanned += 1
                try:
                    prod = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = prod.get("parent_asin") if isinstance(prod, dict) else None
                if asin is None or asin in seen_asins:   # 去重
                    continue
                if not product_valid(prod):              # 过滤空字段
                    continue
                seen_asins.add(asin)
                pool.append(prod)
                if len(pool) >= target * OVERSAMPLE:
                    break

    if len(pool) > target:
        sampled = rng.sample(pool, target)
    else:
        sampled = pool
    print(f"[{category}] scanned {scanned} -> valid {len(pool)} -> sampled {len(sampled)}")
    return sampled


def main():
    rng = random.Random(RANDOM_SEED)
    seen_asins = set()
    kb_chunks = []

    for cat in CATEGORIES:
        prods = load_category(cat, PER_CATEGORY, seen_asins, rng)
        chunks = [product_to_chunk(p) for p in prods]
        kb_chunks.extend(chunks)
        print(f"[{cat}] +{len(chunks)} chunks (cumulative {len(kb_chunks)})")

    # 输出知识库文件
    with open(os.path.join(DATA_DIR, "kb.json"), "w", encoding="utf-8") as f:
        json.dump(kb_chunks, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "kb_pyserini.jsonl"), "w", encoding="utf-8") as f:
        for chunk in kb_chunks:
            out = json.dumps({"id": chunk["id"], "contents": chunk["text"]}, ensure_ascii=False)
            f.write(out + "\n")

    # 统计单词数，用于作业报告
    total_words = sum(len(c["text"].split()) for c in kb_chunks)
    print(f"✅ Corpus built. Documents: {len(kb_chunks)}, total words: {total_words}")


if __name__ == "__main__":
    main()
