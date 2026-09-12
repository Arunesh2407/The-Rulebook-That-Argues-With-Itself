import os
import sys
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.parser import parse_all_documents

class Retriever:
    def __init__(self, collection_name: str = "rulebook_index", persist_dir: str = ".chroma_db"):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def build_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False):
        if force_rebuild or self.collection.count() == 0:
            if self.collection.count() > 0:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn
                )
            
            ids = [c["chunk_id"] for c in chunks]
            documents = [c["content"] for c in chunks]
            metadatas = []
            for c in chunks:
                meta = {
                    "file": c["file"],
                    "section": c["section"],
                    "page": c["page"] if c["page"] is not None else -1
                }
                metadatas.append(meta)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved_chunks = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            ids = results["ids"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, cid, meta, dist in zip(docs, ids, metas, distances):
                page_val = meta.get("page")
                if page_val == -1 or page_val is None:
                    page_val = None
                
                retrieved_chunks.append({
                    "chunk_id": cid,
                    "file": meta.get("file", ""),
                    "section": meta.get("section", ""),
                    "page": page_val,
                    "content": doc,
                    "score": float(dist)
                })

        return retrieved_chunks

if __name__ == "__main__":
    chunks = parse_all_documents()
    retriever = Retriever()
    retriever.build_index(chunks, force_rebuild=True)
    
    query = "Who controls the official academic record when an informal advising comment differs from registrar records?"
    res = retriever.retrieve(query, top_k=5)
    print(f"Retrieved {len(res)} chunks for query: '{query}'")
    for r in res:
        print(f"[{r['file']} | {r['section']}] {r['content'][:150]}...")
