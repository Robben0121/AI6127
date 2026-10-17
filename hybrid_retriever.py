import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import time
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sparse_bm25 import bm25_retrieve

FAISS_INDEX_PATH = "faiss_index.bin"
ID_MAP_PATH = "id_map.json"
KB_PATH = "kb.json"

# load faiss
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
    faiss_id_map = json.load(f)
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb_data = json.load(f)
kb_dict = {item["id"]: item["text"] for item in kb_data}

emb_model = SentenceTransformer("all-MiniLM-L6-v2")

def dense_retrieve(query: str, top_k=20):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    _, I = faiss_index.search(q_emb, k=top_k)
    res = []
    for pos, idx in enumerate(I[0]):
        docid = faiss_id_map[idx]
        res.append((docid, pos))
    return res

def reciprocal_rank_fusion(rank_lists, k=60):
    scores = {}
    for rlist in rank_lists:
        for docid, pos in rlist:
            s = 1.0 / (k + pos + 1)
            scores[docid] = scores.get(docid,0.0)+s
    sorted_docs = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return [docid for docid,_ in sorted_docs]

def hybrid_search(query: str, top_k=10):
    t0 = time.perf_counter()
    bm25_raw = bm25_retrieve(query, top_k=20)
    list_bm25 = [(did,pos) for did,pos,_ in bm25_raw]
    list_dense = dense_retrieve(query, top_k=20)
    fused_ids = reciprocal_rank_fusion([list_bm25, list_dense])
    cost_ms = (time.perf_counter()-t0)*1000
    out = []
    for asin in fused_ids[:top_k]:
        text = kb_dict.get(asin,"")
        out.append({"parent_asin":asin,"passage":text})
    return out, round(cost_ms,2)

if __name__ == "__main__":
    res, ms = hybrid_search("waterproof men's boots", top_k=5)
    print(f"latency {ms} ms")
    for item in res:
        print(item["parent_asin"])
