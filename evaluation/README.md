# 检索与引用评估

## 数据与边界

`nist_questions.json` 的第 2 版包含 20 条人工编写的英文问题，均针对仓库自带的 `NIST_AI_RMF_1.0.pdf`。每题通过 `evidence_groups` 标注必需证据；当前每题只有一组，即一段可核对的原文和 PDF 页码。程序先校验原文在 PDF 和索引中存在，再把包含同一段原文的重叠 Chunk ID 放进**同一证据组**。找到任意一个副本就视为找到该段证据；多个副本不会重复计分。

问题主要涉及框架术语和定义，样本规模小，且只覆盖一份英文文档。结果不代表中文年报、跨文档问题或回答质量。

## 运行

在仓库根目录安装 `requirements.txt`、准备两个 BGE 模型后执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m evaluation.run_retrieval
```

第二条命令会更新 `evaluation/results.json`。它只检索 NIST 文档，即使本地 `data/documents/` 中另有私人 PDF，也不会进入候选集。首次运行会计算这份文档的向量；已缓存的向量会复用。

四组方法为 BM25、Dense、加权 RRF 融合、融合后 BGE Reranker 重排。每组评估前 5 条；Reranker 的输入是融合前 20 个候选。

- **证据命中题数 / Hit@5：**前 5 条中有任意一个 Chunk 命中至少一组证据，该题记为命中。当前每题一组，这也是最直观的主要结果。
- **Evidence Recall@5：**前 5 条覆盖的不同证据组数，除以该题标注的组数。当前每题一组，所以逐题值是 0 或 1；它不再随重叠 Chunk 副本数变化。
- **MRR@5：**首个命中证据组的结果排在第几名，取倒数；未命中记 0，再对所有题取均值。
- **Evidence nDCG@5：**对每组证据只取首次命中的排名，按 `1 / log2(rank + 1)` 折损后除以证据组总数。理想状态是每组都在第 1 名被覆盖，因此分数在 0 到 1 之间。这是按证据组定义的折损覆盖率，与传统按相关文档计分的 nDCG 不完全相同。

`results.json` 保存每题的证据组及其可匹配 Chunk ID、四组前 5 排名和各项得分，便于分析失败案例。文件还记录 PDF 与标注集哈希、分块大小和重叠、RRF 参数、候选数、模型名称与缓存修订号，以及主要软件包版本。比较数值时应保持这些条件一致；本次结果仅代表这份文档和这 20 题。

## 引用支持性核查

`audit_citations.py` 接收人工整理的回答结论与引用。输入 JSON 形式如下：

```json
{
  "claims": [
    {
      "claim": "待核对的一项回答结论",
      "citations": [
        {
          "source": "NIST_AI_RMF_1.0.pdf",
          "pdf_page": 27,
          "chunk_id": "doc_3d466833_chunk_0200"
        }
      ],
      "supported": null,
      "review_note": ""
    }
  ]
}
```

将实际回答拆成逐项结论，填入模型给出的引用并运行：

```powershell
.\.venv\Scripts\python.exe -m evaluation.audit_citations review.json --output audit.json
```

脚本核查文件名、页码和 Chunk ID 是否对应，并检查 Chunk 文本是否仍出现在原 PDF 对应页面，在输出中附上该 Chunk 原文。人工阅读结论及原文后，把输入中的 `supported` 填为 `true` 或 `false`，在 `review_note` 记录理由，再运行一次。脚本不会自动断言语义支持性；`null` 代表尚未审阅，不计入人工支持数。

[示例审阅](citation_review_example.json)来自项目的一张实际回答截图。使用者在本地持有官方来源的 BYD 与 CATL 年报时，可以用上述命令复核：本次 3 项结论的 6 处引用均对应原 PDF 页面，人工判定 3 项结论均有证据支持。年报不随仓库分发；这一个回答的审阅结果不作为系统整体引用支持率。
