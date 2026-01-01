import argparse

from models.evaluator import Evaluator

DATASET = "spider"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=["gpt-5.2", "llama-3.3-70B"], default="gpt-5.2")
    parser.add_argument("--db_size", type=str, default="100")
    parser.add_argument("--apply_level_2", action="store_false")
    parser.add_argument("--schema_filter", type=str, choices=["bm25", "dense"], default=None)
    args = parser.parse_args()

    MODEL = args.model
    DB_SIZE = args.db_size
    F_SUFFIX = args.apply_level_2
    SCHEMA_FILTER = args.schema_filter
    
    ev = Evaluator(
        dataset=DATASET, 
        db_size=DB_SIZE, 
        model=MODEL,
        f_suffix=F_SUFFIX, 
        schema_filter=SCHEMA_FILTER
    )

    # calculate scores
    ev.score_sql()
    
    # print overall scores
    ev.analyze_exa()
    
    # print exa on join subset
    ev.analyze_special_cases()
    
            
    