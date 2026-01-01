
import numpy as np
from copy import deepcopy
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils.filter import normalize, table_to_document, table_to_text

class BM25SchemaFilter:
    
    def __init__(self, schema_json):
        self.schema_json = schema_json
        self.table_names = []
        self.documents = []

        for table_name, table_def in schema_json["schema"].items():
            self.table_names.append(table_name)
            self.documents.append(
                table_to_document(table_name, table_def)
            )

        self.bm25 = BM25Okapi(self.documents)

    def filter(self, question: str, top_k: int = 5):
        query_tokens = normalize(question)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.table_names, scores),
            key=lambda x: x[1],
            reverse=True
        )

        selected_tables = {
            name for name, _ in ranked[:top_k]
        }

        return self._compress_schema(selected_tables)

    # keep only selected tables and valid foreign keys
    def _compress_schema(self, selected_tables):
        compressed = deepcopy(self.schema_json)
        compressed["schema"] = {
            t: compressed["schema"][t]
            for t in selected_tables
        }

        # remove foreign keys pointing to removed tables
        for table in compressed["schema"].values():
            table["foreign_keys"] = [
                fk for fk in table.get("foreign_keys", [])
                if fk["sourceTable"] in selected_tables
            ]

        return compressed



class DenseSchemaFilter:

    def __init__(self, schema_json, model_name="all-MiniLM-L6-v2"):
        self.schema_json = schema_json
        self.table_names = []
        self.table_texts = []

        for table_name, table_def in schema_json["schema"].items():
            self.table_names.append(table_name)
            self.table_texts.append(
                table_to_text(table_name, table_def)
            )

        # Load embedding model (CPU-friendly)
        self.model = SentenceTransformer(model_name)

        # Embed schema ONCE
        self.table_embeddings = self.model.encode(
            self.table_texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def filter(self, question: str, top_k: int = 5):
        query_embedding = self.model.encode(
            question,
            normalize_embeddings=True
        )

        # cosine similarity via dot product (normalized vectors)
        scores = np.dot(self.table_embeddings, query_embedding)

        ranked = sorted(
            zip(self.table_names, scores),
            key=lambda x: x[1],
            reverse=True
        )

        selected_tables = {
            name for name, _ in ranked[:top_k]
        }

        return self._compress_schema(selected_tables)

    def _compress_schema(self, selected_tables):
        compressed = deepcopy(self.schema_json)
        compressed["schema"] = {
            t: compressed["schema"][t]
            for t in selected_tables
        }

        for table in compressed["schema"].values():
            table["foreign_keys"] = [
                fk for fk in table.get("foreign_keys", [])
                if fk["sourceTable"] in selected_tables
            ]

        return compressed

