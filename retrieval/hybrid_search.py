# 混合检索

DEFAULT_BM25_WEIGHT = 1.5
DEFAULT_DENSE_WEIGHT = 1.0

# RRF 倒数排名融合
def reciprocal_rank_fusion(bm25_results,dense_results,rrf_k=60,
                bm25_weight=DEFAULT_BM25_WEIGHT,dense_weight=DEFAULT_DENSE_WEIGHT,
):

    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")
    if bm25_weight <= 0 or dense_weight <= 0:
        raise ValueError("检索器权重必须大于 0")

    fused_results = {}
    for rank, item in enumerate(bm25_results, start=1):
        chunk_id = item["chunk_id"]
        # RRF 公式：1 / (rrf_k + rank)
        fused_results[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": item["document_id"],
            "page": item["page"],
            "text": item["text"],
            # 权重越大，这一路检索结果对最终排名的影响越大。
            "rrf_score": bm25_weight / (rrf_k + rank),
            # 记录它在 BM25 中的排名
            "bm25_rank": rank,
            "dense_rank": None,
        }

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": item["document_id"],
                "page": item["page"],
                "text": item["text"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "dense_rank": None,
            }
        fused_results[chunk_id]["rrf_score"] += dense_weight / (rrf_k + rank)
        fused_results[chunk_id]["dense_rank"] = rank

    result_list = list(fused_results.values())
    #按照score从大到小排序，相同则按照id大小排序
    sorted_results = sorted(result_list,key=lambda result: (-result["rrf_score"], result["chunk_id"]),)

    return sorted_results

class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
    def search(self,query,top_k=10,candidate_k=20,rrf_k=60,
        bm25_weight=DEFAULT_BM25_WEIGHT,dense_weight=DEFAULT_DENSE_WEIGHT,
    ):
        """
        执行一次混合检索。
        参数：
        query：用户的查询文本。
        top_k：最终返回多少个融合结果。
        candidate_k：BM25 和 Dense 各自召回多少个候选。
        rrf_k：RRF 公式中的平滑常数。
        bm25_weight：BM25 排名分数的权重。
        dense_weight：Dense 排名分数的权重。
        """
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if candidate_k < top_k:
            raise ValueError("candidate_k 不能小于 top_k")
        # 调用 BM25Retriever.search()，得到稀疏排序的召回结果。
        bm25_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )
        # 调用 DenseRetriever.search()，得到稠密排序的召回结果。
        dense_results = self.dense.search(
            query=query,
            top_k=candidate_k,
        )
        # 把两路候选交给已经测试通过的 RRF 函数，得到混合检索结果
        fused_results = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            dense_results=dense_results,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
        return fused_results[:top_k]