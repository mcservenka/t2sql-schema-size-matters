import os
import json
import argparse

from configs.paths import SCHEMAS_PATH, SPIDER_DEV_PATH
from models.schema_scaler import SchemaScaler, ScaleConfig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_size", type=int, default=100)
    parser.add_argument("--apply_level_2", action="store_false")
    args = parser.parse_args()

    cfg = ScaleConfig(
        target_total_tables=args.target_size, 
        apply_family_generation=args.apply_level_2,
        apply_join_competition=args.apply_level_2
    )

    with open(SPIDER_DEV_PATH, "r") as f:
        samples = json.load(f)

    databases = set([sample["db_id"] for sample in samples])

    for db in databases:
        with open(f"{SCHEMAS_PATH}spider/{db}.json", "r") as f:
            schema_json = json.load(f)

        sc = SchemaScaler(schema_json = schema_json, cfg=cfg)

        new_schema = sc.enlarge()

        sc.create_new_sqlite_db()

        sc.monitor_metadata()