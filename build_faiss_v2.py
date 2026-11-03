import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8  # 显存小改成4，显存充足可以16
print(f"⚙️  Device: {DEVICE} {'(GPU 加速)' if DEVICE == 'cuda' else '(CPU —— bge-m3 会很慢，请耐心等待或改用 GPU)'}")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KB_PATH = os.path.join(DATA_DIR, "kb.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index_v2.bin")
ID_MAP_PATH = os.path.join(DATA_DIR, "id_map_v2.json")

# 校验知识库文件
if not os.path.exists(KB_PATH):
    raise FileNotFoundError(f"KB 文件不存在：{KB_PATH}，请先运行 build_kb 脚本生成 kb_v2.json")

# 读取知识库
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

texts = [item["text"] for item in kb]
doc_ids = [item["id"] for item in kb]
meta_list = [item["meta"] for item in kb]

# 校验是否存在重复 asin
if len(set(doc_ids)) != len(doc_ids):
    raise ValueError("检测到重复 doc id，上游知识库存在重复商品，请检查 build_kb")

# bge-m3 检索标准前缀！文档和 query 编码必须统一
BGE_SEARCH_PROMPT = "Represent this sentence for searching relevant passages: "
texts_with_prompt = [BGE_SEARCH_PROMPT + t for t in texts]

# 加载模型并移到对应设备
model = SentenceTransformer('BAAI/bge-m3').to(DEVICE)
model.max_seq_length = 8192  # bge-m3 上限

print(f"Encoding {len(texts_with_prompt)} documents ...")
embeddings = model.encode(
    texts_with_prompt,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=BATCH_SIZE,
)
embeddings = embeddings.astype(np.float32)

# IndexFlatIP：归一化向量下 = 余弦相似度，分数越高越相似（推荐）
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# 保存映射：faiss index -> 完整信息，检索直接查表
id_map = {}
for idx, did in enumerate(doc_ids):
    id_map[idx] = {
        "id": did,
        "text": texts[idx],
        "meta": meta_list[idx]
    }

# 持久化
faiss.write_index(index, FAISS_INDEX_PATH)
with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(id_map, f, indent=2, ensure_ascii=False)

print(f"✅ FAISS index saved: {FAISS_INDEX_PATH}")
print(f"✅ ID lookup map saved: {ID_MAP_PATH}")
print(f"✅ Dimension: {dim}, Total vectors: {index.ntotal}")
