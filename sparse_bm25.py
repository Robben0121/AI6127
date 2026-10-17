import json
from rank_bm25 import BM25Okapi

# load knowledge base
with open("kb.json","r",encoding="utf-8") as f:
    kb = json.load(f)

doc_texts = [item["text"] for item in kb]
doc_ids = [item["id"] for item in kb]
id2text = {item["id"]: item["text"] for item in kb}

# tokenize：简单英文分词
tokenized_corpus = [doc.split() for doc in doc_texts]

# build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

def bm25_retrieve(query: str, top_k=20):
    """
    返回 list[(doc_id, rank_position, score)]
    """
    query_tokens = query.split()
    scores = bm25.get_scores(query_tokens)
    # pair score‑id，sort descending
    scored = list(zip(scores, doc_ids))
    scored.sort(key=lambda x:x[0], reverse=True)
    out = []
    for pos, (score, docid) in enumerate(scored[:top_k]):
        out.append((docid, pos, score))
    return out

# quick test
if __name__ == "__main__":
    q = "waterproof men's boots"
    res = bm25_retrieve(q, top_k=5)
    print("==== BM25 test result ====")
    for docid, pos, score in res:
        print(f"docid:{docid}, pos:{pos}, score:{score:.4f}")
