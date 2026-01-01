import os
import json
import time
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from models.prompt import Prompter
from models.schema_builder import SchemaBuilder
from models.schema_filter import BM25SchemaFilter, DenseSchemaFilter
from configs.paths import SPIDER_DEV_PATH, RESULTS_PATH

load_dotenv()

DATASET = "spider"
MODELS = {
    "gpt-5.2": {"provider": "openai", "model": "gpt-5.2"},
    "llama-3.3-70B": {"provider": "together", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
}

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

    os.makedirs(RESULTS_PATH, exist_ok=True)

    # load questions
    with open(SPIDER_DEV_PATH, "r") as f: 
        samples = json.load(f)

    schema_strings = {} # stores schema strings per db_id
    schema_dicts = {} # stores schema objects per db_id
    schema_filters = {} # stores schema filter objects per db_id

    responses = []

    if F_SUFFIX:
        if SCHEMA_FILTER:
            json_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_f_{MODEL}_{SCHEMA_FILTER}_results.json"
            jsonl_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_f_{MODEL}_{SCHEMA_FILTER}_results.jsonl"
        else:
            json_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_f_{MODEL}_results.json"
            jsonl_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_f_{MODEL}_results.jsonl"
    else:
        if SCHEMA_FILTER:
            json_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_{MODEL}_{SCHEMA_FILTER}_results.json"
            jsonl_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_{MODEL}_{SCHEMA_FILTER}_results.jsonl"
        else:
            json_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_{MODEL}_results.json"
            jsonl_path = f"{RESULTS_PATH}{DATASET}_{DB_SIZE}_{MODEL}_results.jsonl"

    # json as main results file
    
    if os.path.exists(json_path):
        raise Exception("Responses already generated.")

    # jsonl as backup    
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                responses.append(json.loads(line))

    # with open(jsonl_path, "w", encoding="utf-8"): pass # create new empty jsonl backup file
    jsonl_out = open(jsonl_path, "a", encoding="utf-8")

    if len(responses) == len(samples):
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(responses, f, indent=4)
        raise Exception("Responses already generated.")

    start_index = len(responses)
    print(f"Starting generating responses at index {start_index}")

    for i, sample in tqdm(enumerate(samples[start_index:], start=start_index)):

        db_id = sample["db_id"]

        if db_id not in schema_strings:
            sb = SchemaBuilder(dataset=DATASET, db_id=db_id, db_size=DB_SIZE, applyChallenges=F_SUFFIX)
            sb.load_schema_json(repopulate_attributes=True)
            schema_dicts[db_id] = sb.schema_object
            schema_strings[db_id] = sb.generate_schema_string(randomize_table_order=True)
        
        if SCHEMA_FILTER:
            if db_id not in schema_filters:
                if SCHEMA_FILTER == "bm25":
                    schema_filters[db_id] = BM25SchemaFilter(schema_json=schema_dicts[db_id])
                elif SCHEMA_FILTER == "dense":
                    schema_filters[db_id] = DenseSchemaFilter(schema_json=schema_dicts[db_id])
                else:
                    raise ValueError("Invalid Schema Filter!")
            
            start_time = time.perf_counter() # start timer
            compressed_schema = schema_filters[db_id].filter(question=sample["question"], top_k=10)
            end_time = time.perf_counter()  # end timer
            filter_duration_seconds = end_time - start_time

            compressed_sb = SchemaBuilder.load_schema_dict(compressed_schema)
            schema_string = compressed_sb.generate_schema_string(randomize_table_order=True)
            tables_included = list(compressed_sb.schema_object["schema"].keys())
            
        else:
            schema_string = schema_strings[db_id]
            filter_duration_seconds = 0
            tables_included = None
        
        p = Prompter(
            provider=MODELS[MODEL]["provider"], model=MODELS[MODEL]["model"], schema_string=schema_string
        )

        # print(f"Generating response {i}")
        response = p.ask_question(question=sample["question"]) # returns llm response dictionary

        response["sql_gold"] = sample["query"]
        response["db_id"] = db_id
        response["filter_duration_seconds"] = filter_duration_seconds
        response["tables_included"] = tables_included # only for filtered schemas (otherwise None)
        response["index"] = i

        responses.append(response)

        jsonl_out.write(json.dumps(response) + "\n")
        jsonl_out.flush()

    jsonl_out.close()

    # create final json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)

    print(f"✅ Results of {DATASET} for size {DB_SIZE} saved to {json_path}")


