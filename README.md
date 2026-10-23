<div align="center">

<img src="static/readme-banner.svg" alt="EvidenceScope：本地 PDF 文档与证据检索主题插图" width="100%">

# EvidenceScope

### 让本地 PDF 研究有据可查

[![Unit tests](https://github.com/Qianying-Chuugu/evidence_scope/actions/workflows/tests.yml/badge.svg)](https://github.com/Qianying-Chuugu/evidence_scope/actions/workflows/tests.yml)

面向本地 PDF 的 AI Agent：由模型调用检索工具，找到相关原文，并在回答中给出可回查的来源。

**上传 PDF → 提出问题 → 检索证据 → 核对原文**

[核心功能](#核心功能) · [快速开始](#快速开始) · [系统架构与技术亮点](#系统架构与技术亮点) · [检索评估](#检索评估) · [当前限制](#当前限制数据隐私与后续方向)

🟢 `Python` &nbsp; 🔵 `FastAPI` &nbsp; 🟣 `DeepSeek API` &nbsp; 🟠 `BM25 + Dense` &nbsp; 🟢 `BGE Reranker`

</div>

> [!IMPORTANT]
> PDF 和索引保存在运行项目的电脑上；问题与检索返回的原文会发送给 DeepSeek API。回答中的引用由模型生成，重要结论请回到原 PDF 核对。

## 演示

<div align="center">
  <img src="static/evidence-scope-demo.gif" alt="实际操作演示：输入问题、调用检索并查看带来源的回答" width="100%">
  <br>
  <sub>真实录屏节选，约 18 秒、2 倍速；年报由使用者本地添加。录屏拍摄于 NIST 示例文件重命名前。</sub>
</div>

<details>
<summary>查看完整页面截图</summary>

<div align="center">
  <img src="static/evidence-scope-demo.png" alt="EvidenceScope 实际网页：提问、PDF 管理和带来源信息的研究结果" width="100%">
  <br>
  <sub>实际运行界面：提问、管理 PDF，并查看带来源信息的回答。截图中的企业年报由使用者本地添加；结论仍需核对原文。</sub>
</div>

</details>

## 核心功能

| 能力 | 实际行为 |
| --- | --- |
| **📁 文档管理** | 在网页上传、查看和删除 PDF；校验类型、大小、可提取文字及重复内容，单文件上限 50 MB |
| **🔄 索引同步** | 为新增或变化的 PDF 更新分块，删除文档时移除索引；向量按文档保存 |
| **🤖 Agent 工具调用** | 模型选择 `search_evidence` 的查询和文档范围，读取结果后可继续检索，最多执行 8 轮 |
| **🔎 混合检索** | BM25 与 Dense Retrieval 分别召回，经加权 RRF 融合、BGE Reranker 重排 |
| **📍 来源追溯** | 检索工具提供原文、文件名、PDF 页码和完整 Chunk ID，供回答引用及人工回查 |

## 快速开始

以下示例适用于 Windows PowerShell。先安装 Python 3.12，并确认下面的命令能找到它；安装依赖和下载模型需要网络连接。

**1. 创建环境并安装依赖**

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果系统没有 `py` 命令，可用已安装的 Python 3.12 解释器运行 `-m venv .venv`。若已有可用的 Python 3.12 虚拟环境，可跳过创建步骤，并用该环境的 `python` 安装依赖和执行后续命令。

**2. 配置 DeepSeek API Key**

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填入自己的 `DEEPSEEK_API_KEY`。该文件已由 `.gitignore` 排除，请勿提交。

**3. 下载模型并启动**

Reranker 需要已缓存的模型才能运行：

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5'); snapshot_download('BAAI/bge-reranker-base')"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

> [!NOTE]
> **首次运行需要等待：**安装命令会下载 Python 依赖；模型命令会从 Hugging Face 下载 `BAAI/bge-small-zh-v1.5` 和 `BAAI/bge-reranker-base`，并占用额外磁盘空间。首次同步 PDF 索引、生成向量或加载模型时，响应可能比后续查询慢。若模型下载失败，请先查看 PowerShell 中的报错，检查是否能访问 Hugging Face，以及网络代理和模型缓存目录的读写权限；修复后重新运行模型下载命令。磁盘占用取决于实际下载的文件，本项目未提供固定数值。

浏览器打开 <http://127.0.0.1:8000/>。仓库自带 `NIST_AI_RMF_1.0.pdf`（[NIST 原始出版页](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)），无需上传文件即可体验；也可以上传自己的可提取文字的 PDF。命令行入口为 `.\.venv\Scripts\python.exe main.py`。

如需体验年报对比，请自行从官方披露渠道获取 [比亚迪 2025 年年度报告](https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-03-28/f5cdbdbf-b138-4e83-8c6d-5a7eecb9e670.PDF) 和 [宁德时代 2025 年年度报告](https://disc.static.szse.cn/disc/disk03/finalpage/2026-03-10/f019f96b-b960-410b-b88f-08838a36d81b.PDF)，在网页上传，或分别保存为 `data/documents/BYD_2025.pdf`、`data/documents/CATL_2025.pdf`。文档目录变化后会同步索引。两份年报不随仓库分发。

### 示例问题

可以直接复制到网页中试用：

- 根据 `NIST_AI_RMF_1.0.pdf`，GOVERN 为什么是跨领域功能？请列出依据所在的 PDF 页码和 Chunk ID。
- 添加两份年报后：比较 `BYD_2025.pdf` 与 `CATL_2025.pdf` 中披露的 2025 年营业收入，并分别给出原文位置。
- 添加宁德时代年报后：在 `CATL_2025.pdf` 中查找营业收入同比变化的依据，返回相关页码和 Chunk ID。

## 系统架构与技术亮点

### 系统架构

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

### 技术亮点

#### 01 · 🤖 Agent 管理检索过程

`AgentHarness` 发起模型调用、执行模型请求的工具、把结果加入对话，并控制迭代次数。文档目录帮助模型选择文件；检索到的原文才是回答事实问题的依据。

#### 02 · 🔎 两条召回路径互补

BM25 适合匹配名称、数字和术语等明确词汇；基于 BGE Embedding 的 Dense Retrieval 用语义相似度寻找不同表述的片段。两路结果共同提供候选，减少对单一匹配方式的依赖。

#### 03 · 🎯 融合与重排分工明确

BM25 与 Dense 的原始分数不直接相加；加权 RRF 根据各自排名合并候选，随后 BGE Reranker 同时读取问题与候选原文，重新排序相关片段。当前权重为 BM25 `1.5`、Dense `1.0`，不代表对所有文档都最优。

#### 04 · 📍 文档变化可同步，证据可定位

`IndexManager` 用清单记录文档状态和内容哈希，只重新分块需要更新的文件，并使变化文档的向量失效。文档目录变化也会清除检索缓存。来源、页码与 Chunk ID 有助于定位原文，但回答与证据是否一致仍需人工核对。

## 自动化测试

自动化测试不调用 DeepSeek API，覆盖 RRF 排名、Agent 工具回填与异常、PDF 上传与删除，以及文档变更后的索引和检索缓存失效。GitHub Actions 在推送和 PR 时使用 `requirements-test.txt` 运行同一组测试，不下载检索模型：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 检索评估

### 测什么

项目用仓库自带的 NIST AI RMF PDF 和 [20 道人工编写的英文问题](evaluation/nist_questions.json)比较四种检索方法。每题标注一段目标原文；检索范围仅限该文档的 367 个 Chunk，不调用 DeepSeek API 生成答案。相邻 Chunk 有 150 个字符的重叠：如果多个 Chunk 都包含同一段目标原文，它们只算**同一组证据**，命中任意一个即可。

### 实验结果

“前 5 名证据命中”是 20 题中至少找到了一个目标证据 Chunk 的题数；MRR@5 衡量首个命中结果的排名，越靠前得分越高，未命中记 0。Reranker 对融合检索的前 20 个候选重新排序。

| 检索方法 | 前 5 名证据命中 | MRR@5 |
| --- | ---: | ---: |
| BM25 | 18/20（90%） | 0.7767 |
| Dense Retrieval | 13/20（65%） | 0.5100 |
| BM25 + Dense + RRF | 18/20（90%） | 0.7458 |
| 融合检索 + BGE Reranker | 20/20（100%） | 0.9100 |

- 在这组问题上，BM25 比单独的 Dense 找到更多指定证据。这些问题包含明确的框架术语，结果可能偏向词法匹配。
- RRF 本次没有优于 BM25：两者的证据命中题数相同，RRF 的 MRR@5 略低。
- Reranker 把融合候选中排在前 5 名之外的部分目标证据提前；它重新排序已有候选，并不产生新证据。

这是单份英文 PDF、20 道题的探索性结果。指定原文不穷尽所有可能支持答案的证据；证据命中也不代表生成的答案或引用一定正确。修正后逐题排名、证据组指标、实验配置和复现方法见 [评估说明](evaluation/README.md)与[结果文件](evaluation/results.json)。

## 引用核查

`audit_citations.py` 核对文件名、PDF 页码和 Chunk ID，并确认 Chunk 文本仍出现在原 PDF 对应页面；“原文是否支持该结论”由人工逐项标注。对一张实际回答截图中的 3 项结论做了[示例审阅](evaluation/citation_review_example.json)：6 处引用均对应原 PDF 页面，3 项结论均获得所列证据支持。该样本依赖使用者本地另行获取的两份年报，且不能代表系统整体引用正确率。

## 项目目录与技术栈

### 项目目录

```text
evidence_scope/
├── api.py                    # FastAPI 页面、文档管理和问答接口
├── main.py                   # Agent 组装、系统提示词与命令行入口
├── document_management.py    # PDF 上传、校验、删除
├── agent/                    # AgentHarness 与工具注册
├── tools/evidence_search.py  # 证据检索工具
├── retrieval/                # 分块、索引、BM25、Dense、RRF、重排
├── tests/                    # 无需模型 API 的自动化测试
├── evaluation/               # 标注问题、检索对比与引用核查工具
├── .github/workflows/tests.yml # 推送和 PR 时运行单元测试
├── static/                   # 网页 HTML、CSS、JavaScript
├── data/documents/           # 本地 PDF；仓库仅附带 NIST AI RMF 示例
├── data/indexes/             # 运行时生成的索引，内容不提交
├── requirements.txt          # Python 依赖
├── requirements-test.txt     # CI 使用的轻量测试依赖
├── LICENSE                   # 项目代码的 MIT 许可证
└── .env.example              # API Key 配置示例
```

### 技术栈

| 技术 | 用途 |
| --- | --- |
| FastAPI、HTML/CSS/JavaScript | 本地网页与 HTTP 接口 |
| OpenAI 兼容 SDK、DeepSeek API | 模型调用与工具决策 |
| PyMuPDF | PDF 文本提取与分页分块 |
| jieba、rank-bm25 | 中文分词与关键词召回 |
| Sentence Transformers、BGE Embedding | 语义向量召回 |
| 加权 RRF、BGE Reranker | 候选融合与重排 |
| JSON、NumPy | 文档清单、分块和向量存储 |

## 当前限制、数据隐私与后续方向

- PDF 和索引保存在本机；问题与检索出的原文会发送给 DeepSeek API。上传敏感资料前请先确认是否适合发送相关内容。
- 项目按本地单机使用设计，启动示例仅监听 `127.0.0.1`；没有账号、用户隔离或公网部署方案。调用 DeepSeek API 需要使用者自己的密钥，并可能产生费用。
- PDF 需能提取足够文字；目前没有 OCR。复杂表格的提取顺序可能影响检索，过短的页面不会进入索引。
- 文件名、页码和 Chunk ID 来自检索结果，但最终引用由模型生成；位置核查不能替代语义支持性判断，重要结论应人工复核。
- 后续可扩展跨文档评估集，并收集真实回答进行人工引用支持性评审。

## 许可证

项目代码采用 [MIT License](LICENSE)。示例 PDF、第三方模型及依赖分别遵循其原始来源的授权条款。
