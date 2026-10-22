#使用 Cross-Encoder 混合检索结果进行 rerank

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"

def load_reranker_model():
    try:
        return CrossEncoder(
            MODEL_NAME,
            local_files_only=True,
            max_length=512,
        )
    except OSError as error:
        raise RuntimeError(
            "本地没有可用的 Reranker 模型，请先下载 BAAI/bge-reranker-base"
        ) from error

class Reranker:
    def __init__(self):
        self.model = load_reranker_model()
    def rerank(self, query, candidates, top_k=5):
        # 搜寻语句、混合检索得到的结果、返回结果数
        query = query.strip()
        if not query:
            raise ValueError("query 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not candidates:
            return []

        # Cross-Encoder 的每个输入都是一个“问题、候选文本”对。
        # 它会同时阅读两段文字，然后直接输出相关性分数。
        query_chunk_pairs = [(query, candidate["text"]) for candidate in candidates]
        # batch_size=8 表示每次处理 8 个文本对。
        scores = self.model.predict(
            query_chunk_pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        reranked_results = []
        # 把候选、原排名和 Reranker 分数合并成一条新结果。
        for original_rank, (candidate, score) in enumerate(zip(candidates, scores),start=1,):
            reranked_item = dict(candidate)
            reranked_item["original_rank"] = original_rank
            reranked_item["reranker_score"] = float(score)
            reranked_results.append(reranked_item)

        reranked_results.sort(key=lambda item: (-item["reranker_score"], item["chunk_id"]))

        return reranked_results[:top_k]
