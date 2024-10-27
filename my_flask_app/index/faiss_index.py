# faiss_index.py

import faiss
import numpy as np

def create_index(model, cases):
    embeddings = [model.encode(case) for case in cases]
    embeddings = np.vstack(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_cases(model, index, cases, query, k=10):
    query_vector = model.encode(query)
    D, I = index.search(query_vector.reshape(1, -1), k)
    return [cases[idx] for idx in I[0]]
