import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 读取知识库
with open("kb.json","r",encoding="utf-8") as f:
    kb = json.load(f)

texts = [item["text"] for item in kb]
doc_ids = [item["id"] for item in kb]

# 加载embedding模型
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, convert_to_numpy=True)

# 构建IndexFlatL2
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# 保存
faiss.write_index(index, "faiss_index.bin")
with open("id_map.json","w",encoding="utf‑8") as f:
    json.dump(doc_ids, f, indent=2)

print("✅ FAISS index saved: faiss_index.bin , id_map.json")
