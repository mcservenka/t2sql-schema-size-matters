import os
import json
import numpy as np
from tqdm import tqdm
from typing import List

from configs.paths import SCHEMAS_PATH, CANDIDATE_PATH
from utils.embeddings import embed_word, cosine_sim


SQL_KEYWORDS = {
    "abort", "action", "add", "after", "all", "alter", "analyze", "and", "as",
    "asc", "attach", "autoincrement", "before", "begin", "between", "by",
    "cascade", "case", "cast", "check", "collate", "column", "commit",
    "conflict", "constraint", "create", "cross", "current_date",
    "current_time", "current_timestamp", "database", "default", "deferrable",
    "deferred", "delete", "desc", "detach", "distinct", "drop", "each", "else",
    "escape", "except", "exclusive", "exists", "explain", "fail", "for",
    "foreign", "from", "full", "glob", "group", "having", "if", "ignore",
    "immediate", "in", "index", "indexed", "initially", "inner", "insert",
    "instead", "intersect", "into", "is", "isnull", "join", "key", "left",
    "like", "limit", "match", "natural", "no", "not", "notnull", "null", "of",
    "offset", "on", "or", "order", "outer", "plan", "pragma", "primary",
    "query", "raise", "recursive", "references", "regexp", "reindex",
    "release", "rename", "replace", "restrict", "right", "rollback", "row",
    "savepoint", "select", "set", "table", "temp", "temporary", "then", "to",
    "transaction", "trigger", "union", "unique", "update", "using", "vacuum",
    "values", "view", "virtual", "when", "where", "with", "without"
}


class NameCasting:

    def __init__(self, embedding_model, dataset:str="spider", db_id:str=None):
        
        self.dataset = dataset
        self.db_id = db_id

        # load schema representations
        self.schema_path = f"{SCHEMAS_PATH}{dataset}/{db_id}.json"
        with open(self.schema_path, "r") as f:
            schema = json.load(f)

        self.schema_original = schema["schema"]
        self.embedding_model = embedding_model

        os.makedirs(CANDIDATE_PATH, exist_ok=True)
        os.makedirs(f"{CANDIDATE_PATH}{dataset}", exist_ok=True)

    # collect origina schema names
    def collect_names(self) -> List[str]:
        names = set()

        # table names
        for table in self.schema_original.keys():
            names.add(table)

        # column names
        for table, table_obj in self.schema_original.items():
            for col in table_obj["columns"]:
                names.add(col["name"])
        
        self.schema_vocab = sorted(names)

        # embeddings of schema vocab
        self.schema_vectors = {
            w: embed_word(w, self.embedding_model)
            for w in self.schema_vocab
            if w in self.embedding_model
        }

        return self.schema_vocab

    # schema centroid based on original schema
    def compute_schema_centroid(self) -> np.ndarray:

        if not self.schema_vocab:
            raise RuntimeError("Run collect_names first to fill schema_vocab")
        
        vectors = []
        for w in self.schema_vocab:
            if w in self.embedding_model:
                vectors.append(embed_word(w, self.embedding_model))
        
        if not vectors:
            raise ValueError("Schema vocab has no embeddable tokens")
        
        self.centroid = np.mean(vectors, axis=0)

        return self.centroid
    
    # generate candidate names
    def build_candidate_pool(
        self,
        noun_vocab: set[str],
        sim_min=0.25,
        sim_max=0.55,
        sim_ambiguity=0.75,
        max_candidates=400,
        min_candidates=200,
        anchor_k=30,
    ):
        # initial centroid
        self.compute_schema_centroid()

        # 🆕 expand anchors if schema is small
        if len(self.schema_vocab) < 30:
            self._expand_schema_anchors(noun_vocab, k=anchor_k)

        # first pass
        candidates = self._collect_candidates(
            noun_vocab,
            sim_min,
            sim_max,
            sim_ambiguity,
        )

        # 🆕 adaptive relaxation
        relax_steps = 0
        while len(candidates) < min_candidates and relax_steps < 4:
            sim_min = max(0.10, sim_min - 0.05)
            sim_max = min(0.90, sim_max + 0.05)

            candidates = self._collect_candidates(
                noun_vocab,
                sim_min,
                sim_max,
                sim_ambiguity,
            )
            relax_steps += 1

        candidates.sort(key=lambda x: -x[1])
        self.candidate_pool = [w for w, _ in candidates[:max_candidates]]

        with open(
            f"{CANDIDATE_PATH}{self.dataset}/{self.db_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.candidate_pool, f, indent=4)

        return self.candidate_pool

    # expand schema semantics in case of small and narrow schema space
    def _expand_schema_anchors(
        self,
        noun_vocab: set[str],
        k: int = 30,
        min_sim: float = 0.15,
    ):
        
        scored = []

        for w in noun_vocab:
            if w in self.schema_vocab:
                continue
            if w not in self.embedding_model:
                continue
            v = embed_word(w, self.embedding_model)
            sim = cosine_sim(v, self.centroid)
            if sim >= min_sim:
                scored.append((w, sim))

        scored.sort(key=lambda x: -x[1])
        extra = [w for w, _ in scored[:k]]

        # add to schema vocab + vectors
        for w in extra:
            self.schema_vocab.append(w)
            self.schema_vectors[w] = embed_word(w, self.embedding_model)

        # recompute centroid
        self.centroid = np.mean(list(self.schema_vectors.values()), axis=0)

        return extra
    
    # collect candidate words
    def _collect_candidates(
        self,
        noun_vocab,
        sim_min,
        sim_max,
        sim_ambiguity,
    ):
        candidates = []

        for w in tqdm(noun_vocab):
            if w in self.schema_vocab:
                continue
            if w in SQL_KEYWORDS:
                continue
            if w not in self.embedding_model:
                continue

            v = embed_word(w, self.embedding_model)
            sim = cosine_sim(v, self.centroid)

            if sim < sim_min or sim > sim_max:
                continue

            # ambiguity check
            ambiguous = False
            for schema_vec in self.schema_vectors.values():
                if cosine_sim(v, schema_vec) > sim_ambiguity:
                    ambiguous = True
                    break

            if not ambiguous:
                candidates.append((w, sim))

        return candidates




