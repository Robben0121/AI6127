# AI6127 课程作业（Amazon 2023 服装数据集）

**NTU AI6127 | 混合检索增强生成系统（Hybrid RAG System）**

- **数据集**：McAuley-Lab Amazon Reviews 2023（`meta_Clothing_Shoes_and_Jewelry.jsonl.gz`）
- **核心实现**：稀疏检索（BM25） + 稠密检索（FAISS） + 倒数秩融合（RRF） + Gradio Web 交互界面

## 项目目录结构

```text
AI6127/
├── dataset.py            # Q1：流式加载 Amazon 商品元数据并采样语料库（10,000 条商品文档）
├── sparse_bm25.py        # BM25 稀疏检索（基于 rank_bm25，无 Pyserini/Java 依赖）
├── build_faiss.py        # 使用 all-MiniLM-L6-v2 与 FAISS 构建稠密向量索引
├── hybrid_retriever.py   # 核心混合检索模块：BM25 + FAISS 检索与 RRF 重排序
├── app_gradio.py         # Q2：Gradio 网页交互界面，支持输入查询并展示检索到的商品段落
├── data/                 # 统一存放所有语料数据与索引文件
│   ├── corpus/           # 采样后的商品 JSONL 语料库（由 dataset.py 生成）
│   └── indexes/
│       └── faiss_index/  # FAISS 向量索引及 Embedding 元数据（由 build_faiss.py 生成）
├── README.md             # 项目说明文档
└── requirements.txt      # Python 依赖环境列表
```

## 环境配置与依赖安装


```bash
pip install -r requirements.txt
```

### `requirements.txt` 依赖列表

```text
datasets
rank-bm25
faiss-cpu
sentence-transformers
gradio
gzip
jsonlines
```

## 问题一（Question 1）：知识库构建

**任务目标**：基于 Amazon 2023 服装（Clothing, Shoes and Jewelry）元数据构建商品知识语料库。

### 构建流程

1. **数据来源**：`meta_Clothing_Shoes_and_Jewelry.jsonl.gz`（[原始数据直链](https://mcauleylab.ucsd.edu:8443/public_datasets/data/amazon_2023/raw/meta_categories/)）
2. **流式加载（Streaming Load）**：以流式方式直接读取压缩的 `.jsonl.gz` 数据，无需将整个庞大数据集完整下载到本地磁盘，从而大幅节省内存与存储空间。
3. **过滤与采样（Filter & Sample）**：采样 10,000 条商品记录，每条记录通过唯一的 `ASIN`（Amazon 标准识别号，即商品唯一 ID）进行标识。
4. **文档格式化（Document Format）**：将每件商品的标题（Title）、类目（Categories）、商品特性（Features）及规格参数拼接为一段完整的文本段落：
   ```json
   {
     "id": "B00IKG5RBS",
     "contents": "Product Title: ...; Categories: ...; Features: ..."
   }
   ```
5. **数据输出（Output）**：将采样处理后的语料库保存至 `data/corpus/` 目录下，供下游检索模块调用。

### 运行 Q1

```bash
# 构建采样后的商品知识语料库
python dataset.py
```

**运行后预期输出**：

```text
✅ Corpus built. Documents: 10000, total words: 1852246
```

## 问题二（Question 2）：混合检索系统与 Web 演示界面

**任务目标**：构建结合稀疏检索与稠密检索的混合检索器（Hybrid Retriever），并搭建交互式 Gradio 网页端演示界面。

**检索流水线（Pipeline）**：BM25 稀疏检索 + FAISS 稠密向量检索 → RRF（倒数秩融合）重排序 → 返回 Top-$k$ 商品文本段落。

### 核心组件说明

| 模块名称 | 技术实现方案 | 核心作用 |
| :--- | :--- | :--- |
| **稀疏检索（BM25）** | 使用纯 Python 库 `rank_bm25`（无需配置 Java / Pyserini，完美兼容 Windows 环境） | 进行精确关键词匹配，擅长捕捉如 `waterproof`、`men's boots` 等特定词项 |
| **稠密检索（FAISS）** | 采用 `all-MiniLM-L6-v2` 向量模型 + `FAISS` 索引（存放在 `data/indexes/faiss_index/`） | 将用户查询与商品段落编码为 384 维稠密向量，实现深层语义相似度检索 |
| **倒数秩融合（RRF）** | 融合 BM25 与 FAISS 各自生成的候选排序列表 | 综合词汇匹配与语义匹配优势，降低单一检索方法的偏差，输出最终排序结果 |
| **Gradio 交互界面** | 基于 `app_gradio.py` 搭建轻量级 Web 页面 | 接收用户输入的商品查询，实时展示检索耗时（Latency）以及包含 ASIN 和商品详情的 Top 检索结果 |

### 运行 Q2

```bash
# 1. 构建 FAISS 稠密向量索引（保存至 data/indexes/faiss_index/）
python build_faiss.py

# 2. 在终端测试混合检索器
python hybrid_retriever.py

# 3. 启动 Gradio 网页端交互应用
python app_gradio.py
```

### 查询与输出示例

```text
Query: waterproof men's boost
Query latency: 44.97 ms
==== Top 5 Retrieved Passages ====
Rank 1 | ASIN: B0BL6WHBNG
Product Title: @ R CORD Mens Hiking Boots Non-Slip Waterproof Hiking Shoes for Men with Ankle Support Durable Mid-Rise Outdoor Boots for Hiking Trekking and Camping; Categories: Clothing, Shoes & Jewelry, Men, Shoes, Outdoor, Hiking & Trekking, Hiking Shoes; Features: LASTING COMFORT - Our waterproof hiking boots have a durable rubber sole that provides greater comfort and stability...
```

## 团队协作说明（Notes for Teammates）

- **Windows 兼容性**：由于 Pyserini 依赖 Java 环境且在 Windows 下容易引发编译报错，本项目跳过 Pyserini，改用纯 Python 的 `rank_bm25` 实现稀疏检索。
- **直接流式读取**：移除了 Hugging Face `datasets` 中已弃用的 `trust_remote_code` 参数，直接通过直链流式读取原始 JSONL 文件。
- **数据集中管理**：所有生成的语料数据（`data/corpus/`）与向量索引（`data/indexes/faiss_index/`）均统一存放在 `data/` 文件夹下。
- **文档唯一标识**：使用 Amazon 元数据中的 `ASIN` 作为每件商品的唯一检索文档 ID（`docid`）。
- **评测集规划**：当前流水线使用采样的 10,000 条商品语料进行检索演示，后续可在此基础上进一步构建专用评测集（Eval Set）。
- **HF Hub 警告说明**：终端若出现关于未认证 Hugging Face Hub Token 的警告，仅为请求速率限制提醒，不影响演示系统的正常运行。

## 报告可扩展方向（Possible Extensions）

- **引入 LLM 生成阶段**：在检索结果基础上接入大语言模型（LLM），根据检索到的商品段落自动生成自然语言回答（完整 RAG 闭环）。
- **批量定量评测**：构建测试查询集，计算 `Recall@k`、`MRR` 等标准信息检索评价指标。
- **消融实验（Ablation Study）**：对比分析仅使用 BM25、仅使用 FAISS 与使用混合 RRF 融合检索在准确率与响应耗时上的表现差异。
