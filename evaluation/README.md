# 检索与引用评估

## 数据与边界

`nist_questions.json` 包含 20 条人工编写的英文问题，均针对仓库自带的 `NIST_AI_RMF_1.0.pdf`。每条记录标注 PDF 页码和一段应能回答该问题的原文短句。运行时先校验短句确实出现在原 PDF 的该页及索引分块中，并把包含短句的所有重叠 Chunk 作为相关结果。因此标签是**指定证据片段**，不穷尽文档中其他可能正确的答案。

问题主要涉及框架术语和定义，样本规模小，且只覆盖一份英文文档。结果不代表中文年报、跨文档问题或回答质量。

## 运行

在仓库根目录安装 `requirements.txt`、准备两个 BGE 模型后执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m evaluation.run_retrieval
```

第二条命令会更新 `evaluation/results.json`。它只检索 NIST 文档，即使本地 `data/documents/` 中另有私人 PDF，也不会进入候选集。首次运行会计算这份文档的向量；已缓存的向量会复用。

四组方法为 BM25、Dense、加权 RRF 融合、融合后 BGE Reranker 重排。每组评估前 5 条；Reranker 的输入是融合前 20 个候选。

- **Recall@5：**前 5 条命中的已标注 Chunk 数，除以该题的已标注 Chunk 总数，再对 20 题取均值。
- **MRR@5：**第一个已标注 Chunk 的倒数排名，未命中记 0，再取均值。
- **nDCG@5：**把含标注短句的 Chunk 视为二元相关，计算折损累计增益并按理想排序归一化。

`results.json` 同时保存每题的相关 Chunk ID、四组前 5 排名和各项得分，便于分析失败案例。比较数值时应保持 PDF、分块规则、模型版本、候选数和标注集一致。

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
