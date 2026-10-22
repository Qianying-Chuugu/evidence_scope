"""供 Agent 调用的 RAG 检索工具。"""

import json
import re

DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 7

_index_manager = None
_embedding_model = None
_retriever_cache = {} #缓存
_reranker = None

_LOCAL_UNIT = re.compile(r"单位\s*[：:]\s*(千元|亿元|元)")
_REPORT_UNIT = re.compile(
    r"(?:财务附注中报表的单位为\s*[：:]\s*|以人民币)(千元|亿元|元)"
)

evidence_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_evidence",
        "description": (
            "在本地 PDF 知识库中检索证据，"
            "返回可用于回答问题的原文、文件名、PDF 页码和 Chunk ID。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "完整、具体的检索查询。按问题需要包含主题、实体、时间、"
                        "术语或其他限定条件，例如：设备手册中的校准步骤和适用条件"
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "可选的 PDF 文件名或相对路径；优先使用文档目录中的 source_path。"
                        "明确目标文件时填写；跨文件比较请分别调用。"
                        "需要发现相关文件或尚不能确定目标时省略，检索全部文件。"
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def _get_index_manager():
    """第一次检索时才导入索引模块。"""
    global _index_manager
    if _index_manager is None:
        from retrieval.index_manager import IndexManager
        _index_manager = IndexManager()
    return _index_manager


def _sync_documents():
    """同步目录，并让后续检索不再使用旧文档的缓存。"""
    index_manager = _get_index_manager()
    previous_documents = index_manager.list_documents()
    documents = index_manager.sync_documents()["documents"]
    if documents != previous_documents:
        _retriever_cache.clear()
    return documents


def get_document_catalog():
    """返回当前索引的精简目录，不加载 PDF、Chunk 或向量。"""
    documents = _sync_documents()
    catalog = []
    for document in sorted(documents.values(), key=lambda item: item["source_path"].casefold()):
        entry = {
            "document_id": document["document_id"],
            "source": document["source"],
            "source_path": document["source_path"],
        }
        for field in ("company", "year"):
            if document.get(field) is not None:
                entry[field] = document[field]
        catalog.append(entry)
    return catalog

def _get_retrieval_models(document_id=None):
    """按文档范围组装并缓存检索器；指定文档时不加载其他索引。"""
    global _embedding_model, _reranker
    from retrieval.bm25_search import BM25Retriever
    from retrieval.dense_search import (
        DenseRetriever,
        encode_chunks,
        load_embedding_model,
    )
    from retrieval.hybrid_search import HybridRetriever
    from retrieval.reranker import Reranker
    index_manager = _get_index_manager()
    if _embedding_model is None:_embedding_model = load_embedding_model()
    # 如果 document_id 有值，就用文档 ID；如果是 None，就用 "__all__" 表示“全部文档”。
    cache_key = document_id or "__all__"
    if cache_key not in _retriever_cache: # 这个文件缓存了吗？没有缓存，则：
        selected_ids = None if document_id is None else [document_id]
        # 检验是否它的向量数据存在，没有则生成
        index_manager.ensure_dense_vectors(lambda chunks: encode_chunks(_embedding_model, chunks),selected_ids,)
        # 把它的分块、向量、文件信息读出来
        snapshot = index_manager.load_snapshot(selected_ids)
        # 预备输入给检索器
        hybrid_retriever = HybridRetriever(
            BM25Retriever(snapshot.chunks),
            DenseRetriever(snapshot.chunks, snapshot.vectors, _embedding_model),
        )
        # 缓存注明
        _retriever_cache[cache_key] = (hybrid_retriever, snapshot.documents)
    if _reranker is None:_reranker = Reranker()
    hybrid_retriever, documents = _retriever_cache[cache_key]
    # 缓存中取出该文件的检索器和文件信息
    return hybrid_retriever, _reranker, documents


def _source_context(chunks, result):
    """从同一文档的现有分块取原文表头和单位声明，不猜测单位。"""
    same_document = [chunk for chunk in chunks if chunk["document_id"] == result["document_id"]]
    result_position = next(
        (index for index, chunk in enumerate(same_document)
         if chunk["chunk_id"] == result["chunk_id"]),
        None,
    )
    if result_position is None:
        return {}
    previous = same_document[:result_position + 1]
    same_page = [chunk for chunk in previous if chunk["page"] == result["page"]]
    context = {}
    if same_page and same_page[0]["chunk_id"] != result["chunk_id"]:
        context["page_header"] = {
            "text": same_page[0]["text"][:200],
            "chunk_id": same_page[0]["chunk_id"],
            "pdf_page": result["page"],
        }

    for chunk in reversed(same_page):
        for line in reversed(chunk["text"].splitlines()):
            match = _LOCAL_UNIT.search(line) or _REPORT_UNIT.search(line)
            if match:
                context["unit_context"] = {
                    "unit": match.group(1), "text": line.strip(),
                    "chunk_id": chunk["chunk_id"], "pdf_page": chunk["page"],
                    "scope": "same_page",
                }
                return context
    for chunk in reversed(previous):
        for line in reversed(chunk["text"].splitlines()):
            match = _REPORT_UNIT.search(line)
            if match:
                context["unit_context"] = {
                    "unit": match.group(1), "text": line.strip(),
                    "chunk_id": chunk["chunk_id"], "pdf_page": chunk["page"],
                    "scope": "report_default",
                }
                return context
    return context

def search_evidence(query: str, top_k: int = DEFAULT_TOP_K, source=None) -> str:

    query = query.strip()
    if not query:
        return json.dumps({"error": "检索查询不能为空"}, ensure_ascii=False)
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        return json.dumps({"error": "top_k 必须是整数"}, ensure_ascii=False)
    if not 1 <= top_k <= 10:
        return json.dumps({"error": "top_k 必须在 1 到 10 之间"}, ensure_ascii=False)
    all_documents = _sync_documents()
    index_manager = _get_index_manager()

    document_id = None
    # 如果指定了文件名，就直接把文件ID找出来
    if source is not None:
        if not isinstance(source, str) or not source.strip():
            return json.dumps({"error": "source 必须是非空文件名"}, ensure_ascii=False)
        source = source.strip()
        try:
            document_id = index_manager.resolve_document_id(source)
        except (KeyError, ValueError) as error:
            available_sources = sorted({document["source"] for document in all_documents.values()})
            return json.dumps(
                {
                    "error": error.args[0],
                    "available_sources": available_sources,
                },
                ensure_ascii=False,
            )
    # 提取指定ID文件数据，如果没有指定，那就把所有文件数据找出来
    hybrid_retriever, reranker, selected_documents = _get_retrieval_models(document_id)
    # HybridRetriever 内部会分别运行 BM25 和 Dense，再用加权 RRF 融合。
    hybrid_candidates = hybrid_retriever.search(
        query=query,
        top_k=DEFAULT_CANDIDATE_K,
        candidate_k=DEFAULT_CANDIDATE_K,
    )
    # Cross-Encoder 阅读查询和每个候选全文，输出最终证据排序。
    reranked_results = reranker.rerank(
        query=query,
        candidates=hybrid_candidates,
        top_k=top_k,
    )
    source_chunks = index_manager.load_all_chunks(
        None if document_id is None else [document_id]
    )
    results = []
    for final_rank, item in enumerate(reranked_results, start=1):
        document = selected_documents[item["document_id"]]
        result = {
                "rank": final_rank,
                "chunk_id": item["chunk_id"],
                "source": document["source"],
                "source_path": document["source_path"],
                "pdf_page": item["page"],
                "text": item["text"],
                "reranker_score": item["reranker_score"],
                "hybrid_rank": item["original_rank"],
                "bm25_rank": item.get("bm25_rank"),
                "dense_rank": item.get("dense_rank"),
            }
        result.update(_source_context(source_chunks, item))
        results.append(result)
    return json.dumps(
        {
            "query": query,
            "results": results,
        },
        ensure_ascii=False,
    )
