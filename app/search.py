from pyserini.encode import AutoQueryEncoder
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import json


class InContextSearcher:
    def __init__(self):

        self.dense_encoder = AutoQueryEncoder('BAAI/bge-base-zh')
        self.dense_searcher = FaissSearcher('data/indexes/bge_index', self.dense_encoder)

        self.sparse_searcher = LuceneSearcher('data/indexes/bm25_zh')
        self.sparse_searcher.set_language('zh')


        self.hybrid_searcher = HybridSearcher(self.dense_searcher, self.sparse_searcher)

    def search(self, query, k=30, alpha=0.6, final_k=5, w=0.7):
 
        hits = self.hybrid_searcher.search(query, k=k, alpha=alpha)

        doc_texts = []
        doc_ids = []

        for hit in hits:
            raw = json.loads(self.sparse_searcher.doc(hit.docid).raw())
            doc_texts.append(raw['contents'])
            doc_ids.append(hit.docid)

        N = len(doc_texts)  
        D = 768            
        
        corpus_vecs = np.array(self.dense_encoder.encode(doc_texts)).reshape(N, D)
        query_vec = np.array(self.dense_encoder.encode([query])).reshape(1, -1)

        selected_indices = InContextSearcher.der_rerank(corpus_vecs, query_vec, final_k, w)

        return [doc_ids[idx] for idx in selected_indices], [doc_texts[idx] for idx in selected_indices]

    @staticmethod
    def der_rerank(corpus_vecs, query_vec, k, w):
        sim_q = cosine_similarity(query_vec, corpus_vecs)[0]         # relevance 得分 (1, N)
        sim_dd = cosine_similarity(corpus_vecs, corpus_vecs)         # 所有候选之间的相似度 (N, N)
        
        selected = [int(np.argmax(sim_q))]  # 首先选出与 query 最相似的样本
        candidates = set(range(len(corpus_vecs))) - set(selected)

        while len(selected) < k and candidates:
            prev_idx = selected[-1]
            candidate_list = list(candidates)

            # 与上一个已选项的 cosine 距离（注意是 1 - 相似度）
            diversity_scores = 1 - sim_dd[candidate_list, prev_idx]
            relevance_scores = sim_q[candidate_list]

            # 最终加权得分
            final_scores = w * diversity_scores + (1 - w) * relevance_scores
            next_idx = candidate_list[int(np.argmax(final_scores))]

            selected.append(next_idx)
            candidates.remove(next_idx)

        return selected












