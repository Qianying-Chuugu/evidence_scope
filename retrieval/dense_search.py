# 稠密检索

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 检验本地是否存在嵌入模型，不存在则下载
def load_embedding_model():
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError:
        print("本地未找到 Embedding 模型，首次运行将从 Hugging Face 下载。")
        return SentenceTransformer(MODEL_NAME)

# 返回二维数组，存储每个chunk的dense向量数据
def encode_chunks(model, chunks):
    # chunks不存在，仍然返回0 x n行的数组
    if not chunks:
        dimension = model.get_sentence_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32)
    return model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

class DenseRetriever:
    def __init__(self, chunks, vectors, model):
        # 分块数据、dense向量数据、模型名称
        if not chunks:
            raise ValueError("chunks 不能为空")
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError("vectors 必须是二维 NumPy 数组")
        if len(chunks) != len(vectors):
            raise ValueError("Chunk 与向量数量不一致")
        self.chunks = chunks
        self.vectors = vectors
        self.model = model

    def search(self, query, top_k=10):
        # 查询语句字符串、返回排名数量

        # 查询语句计算dense向量，计算相似度
        query_vector = self.model.encode(query, normalize_embeddings=True)
        scores = self.vectors @ query_vector
        # 返回从大到小排列下原来下标的列表
        indices = np.argsort(scores)[::-1]

        results = []
        for index in indices:
            chunk = self.chunks[index]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "page": chunk["page"],
                    "score": float(scores[index]),
                    "text": chunk["text"],
                }
            )
            if len(results) >= top_k:
                break
        return results

