<div align="center">

<img src="static/readme-banner.svg" alt="EvidenceScope：本地 PDF 文档与证据检索主题插图" width="100%">

# EvidenceScope

### 让本地 PDF 研究有据可查

面向本地 PDF 的 AI Agent：由模型调用检索工具，找到相关原文，并在回答中给出可回查的来源。

**上传 PDF → 提出问题 → 检索证据 → 核对原文**

[核心功能](#核心功能) · [技术亮点](#技术亮点) · [系统架构](#系统架构) · [快速开始](#快速开始) · [当前限制](#当前限制与后续方向)

🟢 `Python` &nbsp; 🔵 `FastAPI` &nbsp; 🟣 `DeepSeek API` &nbsp; 🟠 `BM25 + Dense` &nbsp; 🟢 `BGE Reranker`

</div>

> [!IMPORTANT]
> PDF 和索引保存在运行项目的电脑上；问题与检索返回的原文会发送给 DeepSeek API。回答中的引用由模型生成，重要结论请回到原 PDF 核对。

## 核心功能

| 能力 | 实际行为 |
| --- | --- |
| **📁 文档管理** | 在网页上传、查看和删除 PDF；校验类型、大小、可提取文字及重复内容，单文件上限 50 MB |
| **🔄 索引同步** | 为新增或变化的 PDF 更新分块，删除文档时移除索引；向量按文档保存 |
| **🤖 Agent 工具调用** | 模型选择 `search_evidence` 的查询和文档范围，读取结果后可继续检索，最多执行 8 轮 |
| **🔎 混合检索** | BM25 与 Dense Retrieval 分别召回，经加权 RRF 融合、BGE Reranker 重排 |
| **📍 来源追溯** | 检索工具提供原文、文件名、PDF 页码和完整 Chunk ID，供回答引用及人工回查 |

## 技术亮点

### 01 · 🤖 Agent 管理检索过程

`AgentHarness` 发起模型调用、执行模型请求的工具、把结果加入对话，并控制迭代次数。文档目录帮助模型选择文件；检索到的原文才是回答事实问题的依据。

### 02 · 🔎 两条召回路径互补

BM25 适合匹配名称、数字和术语等明确词汇；基于 BGE Embedding 的 Dense Retrieval 用语义相似度寻找不同表述的片段。两路结果共同提供候选，减少对单一匹配方式的依赖。

### 03 · 🎯 融合与重排分工明确

BM25 与 Dense 的原始分数不直接相加；加权 RRF 根据各自排名合并候选，随后 BGE Reranker 同时读取问题与候选原文，重新排序相关片段。当前权重为 BM25 `1.5`、Dense `1.0`，不代表对所有文档都最优。

### 04 · 📍 文档变化可同步，证据可定位

`IndexManager` 用清单记录文档状态，只重新分块需要更新的文件，并使变化文档的向量失效。文档目录变化也会清除检索缓存。来源、页码与 Chunk ID 有助于定位原文，但回答与证据是否一致仍需人工核对。

## 系统架构

```mermaid
flowchart LR
    WEB[Web 前端] --> API[FastAPI]
    API -->|提问| AGENT[AgentHarness]
    AGENT <-->|模型决策与工具调用| LLM[DeepSeek API]
    AGENT -->|执行工具| SEARCH[search_evidence]
    SEARCH --> BM25[BM25]
    SEARCH --> DENSE[Dense Retrieval]
    BM25 --> RRF[加权 RRF]
    DENSE --> RRF
    RRF --> RERANK[BGE Reranker]
    RERANK -->|证据回填| AGENT
    AGENT -->|最终回答| API
    API --> WEB

    API -->|上传、列出、删除 PDF| DOC[DocumentManager]
    DOC --> INDEX[IndexManager]
    SEARCH -->|同步与读取索引| INDEX
    INDEX <--> DATA[(本地 PDF、分块与向量)]
```

模型可在一次提问中多次调用 `search_evidence`。文档管理由 FastAPI 调用 `DocumentManager`；检索工具通过 `IndexManager` 同步和读取本地索引。

## 快速开始

以下命令适用于 Windows PowerShell，建议使用 Python 3.12。安装依赖和下载模型需要网络连接。

**1. 安装依赖并配置密钥**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填入自己的 `DEEPSEEK_API_KEY`。该文件已由 `.gitignore` 排除，请勿提交。

**2. 下载模型并启动**

Reranker 需要已缓存的模型才能运行：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5'); snapshot_download('BAAI/bge-reranker-base')"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/>，上传可提取文字的 PDF 后即可提问。也可运行 `.\.venv\Scripts\python.exe main.py` 使用命令行入口。

## 项目目录

```text
evidence_scope/
├── api.py                    # FastAPI 页面、文档管理和问答接口
├── main.py                   # Agent 组装、系统提示词与命令行入口
├── document_management.py    # PDF 上传、校验、删除
├── agent/                    # AgentHarness 与工具注册
├── tools/evidence_search.py  # 证据检索工具
├── retrieval/                # 分块、索引、BM25、Dense、RRF、重排
├── static/                   # 网页 HTML、CSS、JavaScript
├── data/documents/           # 本地 PDF，当前仓库包含示例文档
├── data/indexes/             # 运行时生成的索引，内容不提交
├── requirements.txt          # Python 依赖
└── .env.example              # API Key 配置示例
```

## 技术栈

| 技术 | 用途 |
| --- | --- |
| FastAPI、HTML/CSS/JavaScript | 本地网页与 HTTP 接口 |
| OpenAI 兼容 SDK、DeepSeek API | 模型调用与工具决策 |
| PyMuPDF | PDF 文本提取与分页分块 |
| jieba、rank-bm25 | 中文分词与关键词召回 |
| Sentence Transformers、BGE Embedding | 语义向量召回 |
| 加权 RRF、BGE Reranker | 候选融合与重排 |
| JSON、NumPy | 文档清单、分块和向量存储 |

## 当前限制与后续方向

- 项目按本地单机使用设计，启动示例仅监听 `127.0.0.1`；没有账号、用户隔离或公网部署方案。调用 DeepSeek API 需要使用者自己的密钥，并可能产生费用。
- PDF 需能提取足够文字；目前没有 OCR。复杂表格的提取顺序可能影响检索，过短的页面不会进入索引。
- 文件名、页码和 Chunk ID 来自检索结果，但最终引用由模型生成；项目尚未程序化验证每项结论是否受到引用原文支持，重要结论应人工复核。
- 后续可增加回答与证据的一致性检查，并用不同类型文档评估检索表现。
