import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Device: {DEVICE} {'(GPU 加速)' if DEVICE == 'cuda' else '(CPU —— bge-m3 会很慢，请耐心等待或改用 GPU)'}")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KB_PATH = os.path.join(DATA_DIR, "kb.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map.json")

# 读取知识库
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

texts = [item["text"] for item in kb]
doc_ids = [item["id"] for item in kb]

# 加载 embedding 模型（bge-m3：8192 token 长上下文、1024 维）
# 注意：模型名需与 hybrid_retriever.py 中保持一致
model = SentenceTransformer('BAAI/bge-m3')
# 归一化嵌入，使 IndexFlatL2 的距离与余弦相似度单调等价
print(f"Encoding {len(texts)} documents ...")
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=16,
)

# 构建IndexFlatL2
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 保存
faiss.write_index(index, FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(doc_ids, f, indent=2)

print(f"✅ FAISS index saved: {FAISS_INDEX_PATH} , {ID_MAP_PATH}")
