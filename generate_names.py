import json
import fasttext
import argparse

from models.name_casting import NameCasting
from utils.vocab import load_noun_vocabulary
from configs.paths import SPIDER_DEV_PATH

EMBEDDING_MODEL = fasttext.load_model("cc.en.300.bin")
NOUN_VOCAB = load_noun_vocabulary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_min", type=float, default=0.55)
    parser.add_argument("--sim_max", type=float, default=0.25)
    parser.add_argument("--sim_ambiguity", type=float, default=0.75)
    parser.add_argument("--min_candidates", type=int, default=400)
    parser.add_argument("--max_candidates", type=int, default=1000)
    parser.add_argument("--anchor_k", type=int, default=30)
    args = parser.parse_args()

    with open(SPIDER_DEV_PATH, "r") as f:
        samples = json.load(f)

    databases = set([sample["db_id"] for sample in samples])

    for db in databases:
        
        # init name casting
        se = NameCasting(
            embedding_model=EMBEDDING_MODEL,
            dataset="spider",
            db_id=db
        )

        # collect original schema names
        object_names = se.collect_names()

        # compute schema centroid
        cen = se.compute_schema_centroid()

        # generate candidate pool
        candidates = se.build_candidate_pool(
            noun_vocab=NOUN_VOCAB,
            sim_min=args.sim_min,
            sim_max=args.sim_max,
            sim_ambiguity=args.sim_ambiguity,
            anchor_k=args.anchor_k,
            max_candidates=args.max_candidates,
            min_candidates=args.min_candidates
        )
        
    print("Candidate Name Generation finished.")