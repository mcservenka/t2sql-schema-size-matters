import os
import copy
import json
import shutil
import random
import sqlite3
from dataclasses import dataclass

from configs.paths import DATASETS_PATH, SCHEMAS_PATH, CANDIDATE_PATH, SPIDER_DATABASE_PATH, METADATA_PATH
from utils.schema import get_original_column_names, make_column, sqlite_create_table_sql

ENTITY_ATTR_TEMPLATES = [
    ("name", "TEXT"),
    ("description", "TEXT"),
    ("content", "TEXT"),
    ("type", "TEXT"),
    ("status", "TEXT"),
    ("created_at", "TIMESTAMP"),
    ("updated_at", "TIMESTAMP"),
    ("count", "INT"),
    ("year", "INT"),
]

META_ATTR_TEMPLATES = [
    ("key", "TEXT"),
    ("value", "TEXT"),
    ("timestamp", "TIMESTAMP"),
    ("flag", "INT"),
    ("status", "TEXT"),
]

@dataclass
class ScaleConfig:
    target_total_tables: int = 100
    ratio_entity: float = 0.60
    ratio_join: float = 0.25
    ratio_meta: float = 0.15

    # level 2 settings
    apply_family_generation:bool = False
    orig_prob: float = 0.4
    apply_join_competition: bool = False

    # columns per table (includes pk)
    entity_min_cols: int = 6
    entity_max_cols: int = 12
    meta_min_cols: int = 5
    meta_max_cols: int = 8

    # foreign-key
    max_anchor_links: int = 5 # maximum fk connections to original tables
    entity_fk_prob: float = 0.30 # chance an entity table gets an fk at all
    meta_fk_prob: float = 0.10

    # determinism
    seed: int = 42

class SchemaScaler:

    def __init__(self, schema_json:dict, cfg: ScaleConfig):
        
        self.dataset = schema_json["dataset"]
        self.db_id = schema_json["db_id"]
        if self.dataset == "spider":
            self.db_path_original = SPIDER_DATABASE_PATH
        
        self.schema_original = schema_json["schema"]
        self.schema_new = copy.deepcopy(self.schema_original) # start with original schema
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        with open(f"{CANDIDATE_PATH}{self.dataset}/{self.db_id}.json", "r") as f:
            self.candidate_words = json.load(f)
        
        self.original_tables = list(self.schema_original.keys())
        self.original_table_set = set(self.original_tables)
        self.original_col_set = get_original_column_names(self.schema_original)

        # track used names
        self.used_table_names = self.original_table_set
        self.anchor_links_used = 0

        # track generated tables
        self.entity_tables = []
        self.meta_tables = []

        # track concept families
        self.entity_families = {}  # base_word -> [table_names]

        # scaled data paths
        if self.cfg.apply_family_generation:
            self.scaled_schemas_path = f"{SCHEMAS_PATH}{self.dataset}_{self.cfg.target_total_tables}_f"
            self.scaled_sqlite_path = f"{DATASETS_PATH}{self.dataset}_{self.cfg.target_total_tables}_f/database/"
            self.meta_dir = f"{METADATA_PATH}{self.dataset}_{self.cfg.target_total_tables}_f/"
        else:
            self.scaled_schemas_path = f"{SCHEMAS_PATH}{self.dataset}_{self.cfg.target_total_tables}"
            self.scaled_sqlite_path = f"{DATASETS_PATH}{self.dataset}_{self.cfg.target_total_tables}/database/"
            self.meta_dir = f"{METADATA_PATH}{self.dataset}_{self.cfg.target_total_tables}/"
        os.makedirs(self.scaled_schemas_path, exist_ok=True)
        os.makedirs(self.scaled_sqlite_path, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

    
    # get next candidate word that doesn't collide with original table names
    def _pop_fresh_word(self):
        while self.candidate_words:
            w = self.candidate_words.pop(0)
            if w in self.used_table_names: # already used
                continue
            if w in self.original_table_set: # in original tables
                continue
            if w in self.original_col_set: # in original columns (not mandatory)
                continue
            self.used_table_names.add(w)
            return w
        raise RuntimeError("Ran out of candidate words. Increase candidate pool size.")
    
    # returns (table, pk_col) for an original anchor table or None if none available.
    def _choose_original_anchor(self):
        if self.anchor_links_used >= self.cfg.max_anchor_links:
            return None

        # choose an original table with a primary key (deterministic)
        for t in sorted(self.original_tables):
            pks = self.schema_original[t].get("primary_keys", [])
            if pks:
                self.anchor_links_used += 1
                return t, pks[0]
        return None
    
    def _entity_pk_name(self, w: str) -> str:
        return f"{w}_id"

    def _make_entity_table(self, table_name: str, base_original: str = None) -> None:
        
        pk = self._entity_pk_name(table_name)

        # deterministic number of columns
        n_cols = self.rng.randint(self.cfg.entity_min_cols, self.cfg.entity_max_cols)
        # encourage column overlap for family members
        if "_" in table_name:
            n_cols = max(self.cfg.entity_min_cols, n_cols - 1)


        cols = [make_column(pk, "INT", pk=1, notnull=True)]
        pk_list = [pk]
        fk_list = []

        if base_original:
            # level 2 -> reuse subset of original columns
            _, sampled_cols = self._sample_original_columns(base_original, reuse_ratio=0.6)

            for c in sampled_cols:
                cname = c["name"]
                ctype = c["type"]
                if cname == pk:
                    continue
                cols.append(make_column(cname, ctype, pk=0, notnull=False))

        else:
            # level 1 synthetic entity
            n_cols = self.rng.randint(self.cfg.entity_min_cols, self.cfg.entity_max_cols)
            if "_" in table_name:
                n_cols = max(self.cfg.entity_min_cols, n_cols - 1)

            for (attr_name, attr_type) in ENTITY_ATTR_TEMPLATES:
                if len(cols) >= n_cols:
                    break
                if attr_name in {c["name"] for c in cols}:
                    continue
                cols.append(make_column(attr_name, attr_type, pk=0, notnull=False))

            i = 1
            while len(cols) < n_cols:
                cname = f"attr_{i}"
                if cname not in {c["name"] for c in cols}:
                    cols.append(make_column(cname, "TEXT"))
                i += 1

        self.schema_new[table_name] = {
            "columns": cols,
            "primary_keys": pk_list,
            "foreign_keys": fk_list
        }
        self.entity_tables.append(table_name)

    def _make_meta_table(self, base_word: str) -> None:
        # deterministic naming
        suffix = self.rng.choice(["_log", "_entry", "_state", "_snapshot"])
        table_name = f"{base_word}{suffix}"
        if table_name in self.used_table_names:
            # ensure uniqueness if collision
            k = 2
            while f"{table_name}_{k}" in self.used_table_names:
                k += 1
            table_name = f"{table_name}_{k}"
        self.used_table_names.add(table_name)

        pk = f"{base_word}_id"
        n_cols = self.rng.randint(self.cfg.meta_min_cols, self.cfg.meta_max_cols)

        cols = [make_column(pk, "INT", pk=1, notnull=True)]
        pk_list = [pk]
        fk_list = []

        # optional weak fk to a random entity table
        if self.entity_tables and (self.rng.random() < self.cfg.meta_fk_prob):
            target = self.rng.choice(self.entity_tables)
            target_pk = self._entity_pk_name(target)
            fk_col = target_pk
            if fk_col != pk:
                cols.append(make_column(fk_col, "INT"))
                fk_list.append({
                    "sourceTable": target,
                    "sourceColumn": target_pk,
                    "targetColumn": fk_col
                })

        for (attr_name, attr_type) in META_ATTR_TEMPLATES:
            if len(cols) >= n_cols:
                break
            existing = {c["name"] for c in cols}
            if attr_name in existing:
                continue
            cols.append(make_column(attr_name, attr_type))

        i = 1
        while len(cols) < n_cols:
            cname = f"meta_{i}"
            if cname not in {c["name"] for c in cols}:
                cols.append(make_column(cname, "TEXT"))
            i += 1

        self.schema_new[table_name] = {
            "columns": cols,
            "primary_keys": pk_list,
            "foreign_keys": fk_list
        }
        self.meta_tables.append(table_name)
    
    def _make_join_table(self, a: str, b: str) -> None:
        a_, b_ = sorted([a, b])
        table_name = f"{a_}_{b_}"
        if table_name in self.used_table_names:
            return
        self.used_table_names.add(table_name)

        a_pk = self._entity_pk_name(a_)
        b_pk = self._entity_pk_name(b_)

        cols = [
            make_column(a_pk, "INT", pk=1, notnull=True),
            make_column(b_pk, "INT", pk=2, notnull=True),
        ]

        self.schema_new[table_name] = {
            "columns": cols,
            "primary_keys": [a_pk, b_pk],
            "foreign_keys": [
                {"sourceTable": a_, "sourceColumn": a_pk, "targetColumn": a_pk},
                {"sourceTable": b_, "sourceColumn": b_pk, "targetColumn": b_pk},
            ]
        }

    # creates synthetic bridge table between valid join paths
    def _make_join_bridge(self, src: str, tgt: str, src_pk: str, tgt_pk: str):
        
        bridge_name = f"{src}_{tgt}_bridge"
        if bridge_name in self.used_table_names:
            return

        self.used_table_names.add(bridge_name)

        bridge_pk = f"{bridge_name}_id"
        bridge_src_pk = f"{src_pk}_fk1"
        bridge_tgt_pk = f"{tgt_pk}_fk2"

        cols = [
            make_column(bridge_pk, "INT", pk=1, notnull=True),
            make_column(bridge_src_pk, "INT"),
            make_column(bridge_tgt_pk, "INT"),
        ]

        self.schema_new[bridge_name] = {
            "columns": cols,
            "primary_keys": [bridge_pk],
            "foreign_keys": [
                {
                    "sourceTable": src,
                    "sourceColumn": src_pk,
                    "targetColumn": bridge_src_pk,
                },
                {
                    "sourceTable": tgt,
                    "sourceColumn": tgt_pk,
                    "targetColumn": bridge_tgt_pk,
                },
            ],
        }

        self.entity_tables.append(bridge_name)

    # main enlarger
    def enlarge(self) -> dict:
        # compute how many new tables to add
        n_orig = len(self.schema_original)
        n_new = max(0, self.cfg.target_total_tables - n_orig)

        n_entity = int(round(n_new * self.cfg.ratio_entity))
        n_join = int(round(n_new * self.cfg.ratio_join))
        n_meta = n_new - n_entity - n_join  # absorb rounding

        # generate entity tables
        for _ in range(n_entity):
            w = self._pop_fresh_word()

            if self.cfg.apply_family_generation and self.rng.random() < self.cfg.orig_prob:
                base = self.rng.choice(self.original_tables)
                table_name = f"{base}_{w}"
                self._make_entity_table(table_name, base_original=base)
            else:
                self._make_entity_table(w)

        # only for level 2 make join bridge tables
        if self.cfg.apply_join_competition:
            for src in sorted(self.schema_original.keys()):
                for fk in self.schema_original[src].get("foreign_keys", []):
                    tgt = fk["sourceTable"]
                    tgt_pk = fk["sourceColumn"]
                    src_pk = fk["targetColumn"]

                    self._make_join_bridge(src, tgt, src_pk, tgt_pk)
            
        # generate join tables from entity pairs deterministically
        pairs = []
        ents_sorted = sorted(self.entity_tables)
        for i in range(len(ents_sorted)):
            for j in range(i + 1, len(ents_sorted)):
                pairs.append((ents_sorted[i], ents_sorted[j]))        
        self.rng.shuffle(pairs) # deterministically shuffle to diversify while remaining reproducible

        created = 0
        for (a, b) in pairs:
            if created >= n_join:
                break
            before = len(self.schema_new)
            self._make_join_table(a, b)
            after = len(self.schema_new)
            if after > before:
                created += 1

        # generate metadata tables
        for _ in range(n_meta):
            w = self._pop_fresh_word()
            self._make_meta_table(w)

        # store new schema object
        schema = {
            "dataset": self.dataset,
            "db_id": self.db_id,
            "schema": self.schema_new
        }
        with open(
            f"{self.scaled_schemas_path}/{self.db_id}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(schema, f, indent=4)

        return schema

    # create new sqlite databases
    def create_new_sqlite_db(self) -> None:
        target_dir = f"{self.scaled_sqlite_path}{self.db_id}/"
        os.makedirs(target_dir, exist_ok=True)

        # paths
        src_db = f"{self.db_path_original}{self.db_id}/{self.db_id}.sqlite"
        dst_db = f"{target_dir}{self.db_id}.sqlite"

        if os.path.exists(dst_db):
            os.remove(dst_db)
            
        shutil.copy2(src_db, dst_db)

        conn = sqlite3.connect(dst_db)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        original_tables = set(self.schema_original.keys())
        new_tables = [
            t for t in self.schema_new.keys()
            if t not in original_tables
        ]

        # create tables
        for table in new_tables:
            try:
                sql = sqlite_create_table_sql(table, self.schema_new[table])
                cur.execute(sql)
            except Exception as e:
                print(f"Error in Sqlite Creation on {self.dataset}/{self.db_id}")
                print(f"Concerning table {table}")
                raise Exception(e)

        conn.commit()
        conn.close()

    # samples subset of original table columns (always with pk)
    def _sample_original_columns(self, base_table: str, reuse_ratio: float = 0.6):
        
        orig = self.schema_original[base_table]
        orig_cols = orig["columns"]
        orig_pks = set(orig.get("primary_keys", []))

        # separate PK and non-PK columns
        pk_cols = [c for c in orig_cols if c["name"] in orig_pks]
        non_pk_cols = [c for c in orig_cols if c["name"] not in orig_pks]

        k = max(1, int(round(len(non_pk_cols) * reuse_ratio)))
        sampled_non_pk = self.rng.sample(non_pk_cols, min(k, len(non_pk_cols)))

        return pk_cols, sampled_non_pk

    # create metadata files
    def monitor_metadata(self):

        original_tables = set(self.schema_original.keys())
        all_tables = list(self.schema_new.keys())
        synthetic_tables = [t for t in all_tables if t not in original_tables]

        def count_fks(schema: dict) -> int:
            n = 0
            for t in schema:
                n += len(schema[t].get("foreign_keys", []))
            return n

        # table type breakdown (based on naming)
        join_tables = [
            t for t in synthetic_tables
            if ("_" in t) and (t.endswith("_log") is False) and (t.endswith("_entry") is False)
            and (t.endswith("_state") is False) and (t.endswith("_snapshot") is False)
            # this heuristic will count some entity tables with underscores if they exist
        ]
        meta_tables = [
            t for t in synthetic_tables
            if t.endswith("_log") or t.endswith("_entry") or t.endswith("_state") or t.endswith("_snapshot")
        ]
        # entity tables are the rest
        entity_tables = [t for t in synthetic_tables if (t not in join_tables and t not in meta_tables)]

        meta = {
            "dataset": self.dataset,
            "db_id": self.db_id,
            "target_total_tables": self.cfg.target_total_tables,
            "seed": self.cfg.seed,
            "ratios": {
                "entity": self.cfg.ratio_entity,
                "join": self.cfg.ratio_join,
                "meta": self.cfg.ratio_meta,
            },
            "counts": {
                "tables_original": len(original_tables),
                "tables_total": len(all_tables),
                "tables_synthetic": len(synthetic_tables),
                "tables_synthetic_entity_est": len(entity_tables),
                "tables_synthetic_join_est": len(join_tables),
                "tables_synthetic_meta_est": len(meta_tables),
                "foreign_keys_original": count_fks(self.schema_original),
                "foreign_keys_total": count_fks(self.schema_new),
                "foreign_keys_synthetic_only": count_fks({t: self.schema_new[t] for t in synthetic_tables}),
            }
        }

        out_path = f"{self.meta_dir}/{self.db_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)