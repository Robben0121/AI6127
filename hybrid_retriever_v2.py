import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import time
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sparse_bm25 import bm25_retrieve

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index_v2.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map_v2.json")
KB_PATH = os.path.join(DATA_DIR, "kb_v2.json")

# load faiss
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)
# 修复：把字符串key转回int，解决 KeyError
faiss_id_map = {int(k): v for k, v in raw.items()}

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb_data = json.load(f)
kb_dict = {item["id"]: item["text"] for item in kb_data}
kb_meta = {item["id"]: item.get("meta", {}) for item in kb_data}

# 和 build_faiss 保持一致的检索prompt
BGE_SEARCH_PROMPT = "Represent this sentence for searching relevant passages: "
emb_model = SentenceTransformer("BAAI/bge-m3")

def dense_retrieve(query: str, top_k=20):
    # 加上prompt！
    q_emb = emb_model.encode([BGE_SEARCH_PROMPT + query], convert_to_numpy=True, normalize_embeddings=True)
    _, I = faiss_index.search(q_emb, k=top_k)
    res = []
    for pos, idx in enumerate(I[0]):
        if idx == -1:
            continue
        # 这里 faiss_id_map[idx] 拿到的是 {"id":"xxx", "text":"xxx", "meta":{...}}
        record = faiss_id_map[idx]
        docid = record["id"]
        res.append((docid, pos))
    return res

def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for rlist in rank_lists:
        for docid, pos in rlist:
            s = 1.0 / (k + pos + 1)
            scores[docid] = scores.get(docid, 0.0) + s
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docid for docid, _ in sorted_docs]

def hybrid_search(query: str, top_k=10):
    t0 = time.perf_counter()
    bm25_raw = bm25_retrieve(query, top_k=20)
    list_bm25 = [(did, pos) for did, pos, _ in bm25_raw]
    list_dense = dense_retrieve(query, top_k=20)
    fused_ids = reciprocal_rank_fusion([list_bm25, list_dense])
    cost_ms = (time.perf_counter() - t0) * 1000
    out = []
    for asin in fused_ids[:top_k]:
        text = kb_dict.get(asin, "")
        meta = kb_meta.get(asin, {})
        out.append({
            "parent_asin": asin,
            "passage": text,
            "price": meta.get("price"),
            "average_rating": meta.get("average_rating"),
            "rating_number": meta.get("rating_number"),
            "store": meta.get("store"),
        })
    return out, round(cost_ms, 2)

if __name__ == "__main__":
    res, ms = hybrid_search("waterproof men's boots", top_k=5)
    print(f"latency {ms} ms")
    for item in res:
        print(item["parent_asin"])
