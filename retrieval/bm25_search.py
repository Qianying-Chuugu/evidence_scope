# 稀疏检索

import jieba
from rank_bm25 import BM25Okapi

#添加一般的分词词语，防止进一步被拆分，提高检索准确率
jieba.add_word("营业收入")
jieba.add_word("宁德时代")


def tokenize(text):
    """ 中文分词，分词中去除空格 """
    return [word for word in jieba.lcut(text) if word.strip()]

class BM25Retriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("chunks 不能为空")
        self.chunks = chunks
        # 中文分词
        self.documents = [tokenize(chunk["text"]) for chunk in self.chunks]
        # 建立bm25搜索索引
        self.bm25 = BM25Okapi(self.documents)

    def search(self, query, top_k=10):
        # 分块数据、dense向量数据

        # 查询语句进行分词
        query_tokens = list(tokenize(query))
        # 用查询分词进行算分
        scores = self.bm25.get_scores(query_tokens)

        results = []
        for index in scores.argsort()[::-1]:
            chunk = self.chunks[index]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "score": float(scores[index]),
                "page": chunk["page"],
                "text": chunk["text"]
            })
            if len(results) >= top_k:
                break

        return results

