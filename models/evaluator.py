
import os
import re
import json
from tqdm import tqdm

from configs.paths import SCHEMAS_PATH, RESULTS_PATH, SPIDER_DATABASE_PATH
from external.testsuitesqleval.exec_eval import eval_exec_match


TABLE_REF_REGEX = re.compile(
    r"""
    (?<=from|join) # preceded by FROM or JOIN
    \s+                     
    (?:["`\[]?) # optional opening quote
    ([a-zA-Z_][\w]*) # capture table name (group 1)
    (?:["`\]]?) # optional closing quote
    (?:\s*\.\s*
        (?:["`\[]?)
        ([a-zA-Z_][\w]*) # optional schema.table (group 2)
        (?:["`\]]?)
    )?
    """,
    re.IGNORECASE | re.VERBOSE
)


class Evaluator:

    def __init__(self, dataset:str=None, db_size:str=None, model:str=None, f_suffix:bool=True, schema_filter:str=None):

        self.dataset = dataset
        self.db_size = db_size
        self.f_suffix = f_suffix
        self.schema_filter = schema_filter
        self.model = model

        # path definitions
        if db_size == "0":
            if dataset == "spider": 
                self.db_path = SPIDER_DATABASE_PATH
                self.schemas_path = f"{SCHEMAS_PATH}{self.dataset}/"
                self.results_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_{self.model}_results.json"
                self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_{self.model}_eval.json"
            else: raise Exception("Invalid dataset selection.")
        else:
            if self.f_suffix:
                self.db_path = f"data/datasets/{dataset}_{db_size}_f/database/"
                self.schemas_path = f"{SCHEMAS_PATH}{self.dataset}_{db_size}_f/"
                if schema_filter:
                    self.results_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_f_{self.model}_{self.schema_filter}_results.json"
                    self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_f_{self.model}_{self.schema_filter}_eval.json"
                else:
                    self.results_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_f_{self.model}_results.json"
                    self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_f_{self.model}_eval.json"
            else:
                self.db_path = f"data/datasets/{dataset}_{db_size}/database/"
                self.schemas_path = f"{SCHEMAS_PATH}{self.dataset}_{db_size}/"

                if schema_filter:
                    self.results_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_{self.model}_{self.schema_filter}_results.json"
                    self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_{self.model}_{self.schema_filter}_eval.json"
                else:
                    self.results_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_{self.model}_results.json"
                    self.eval_path = f"{RESULTS_PATH}{self.dataset}_{self.db_size}_{self.model}_eval.json"

        with open(self.results_path, "r") as f: 
            self.results = json.load(f)
        
    # generate scores
    def score_sql(self):
        
        if os.path.exists(self.eval_path):
            raise Exception("Evaluation files already generated")

        total_score = 0

        for _, result in tqdm(enumerate(self.results)):

            db_id = result.get("db_id")
            gold_sql = result.get("sql_gold")
            pred_sql = result.get("response", {}).get("sql")
            exec_score = self.execution_accuracy(db_id=db_id, gold_sql=gold_sql, pred_sql=pred_sql)

            total_score += exec_score

            result["execution_accuracy"] = exec_score
        
        with open(self.eval_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

        print(f"EXA for {self.model} in {self.dataset}_{self.db_size}: {total_score / len(self.results)}")
        return total_score / len(self.results)

    # calculate exa
    def execution_accuracy(self, db_id:str, gold_sql:str, pred_sql:str):

        db = f"{self.db_path}{db_id}/{db_id}.sqlite"

        if self.dataset == "spider":
            try:
                exec_score = eval_exec_match(db=db, p_str=pred_sql, g_str=gold_sql, plug_value=False, keep_distinct=True, progress_bar_for_each_datapoint=False)
            except:
                exec_score = 0
        else:
            raise Exception("Uknown dataset during evaluation.")
        
        return exec_score

    # print overall exa
    def analyze_exa(self):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception(f"Create eval file for {self.dataset}_{self.db_size} first.")
        
        exa = 0
        duration = 0
        duration_correct = 0        
        tokens = 0
        tokens_correct = 0
        total_count = 0
        correct_count = 0

        if self.f_suffix:
            level = "L2"
        else:
            level = "L1"

        for sample in eval:
            exa += sample["execution_accuracy"]
            duration += sample["duration_seconds"]
            tokens += sample["total_tokens"]
            if sample.get("filter_duration_seconds"):
                duration += sample["filter_duration_seconds"] # only when filter was applied
            total_count += 1

            if sample["execution_accuracy"] == 1:
                duration_correct += sample["duration_seconds"]
                if sample.get("filter_duration_seconds"):
                    duration_correct += sample["filter_duration_seconds"]
                tokens_correct += sample["total_tokens"]
                correct_count += 1
        
        exa_score = round(exa / total_count * 100, 2)
        duration_avg = round(duration / total_count, 2)
        tokens_avg = round(tokens / total_count, 2)
        duration_correct_avg = round(duration_correct / correct_count, 2)
        tokens_correct_avg = round(tokens_correct / correct_count, 2)
        print(f"{self.dataset} | {self.db_size} | {self.model} | Count: {total_count} | ExA: {exa_score} | Duration: {duration_avg} | Tokens: {tokens_avg}")
        # print(f"{self.dataset}-dev-{self.db_size};{level};{self.schema_filter};{self.model};{self.schema_filter};{total_count};{correct_count};{exa_score};{duration_avg};{duration_correct_avg};{tokens_avg};{tokens_correct_avg}")
        # print(f"{self.dataset}-dev-{self.db_size} | {level} | {self.schema_filter} | {self.model} | Count: {total_count} | ExA: {exa_score} | s/query: {duration_avg} | s/correct: {duration_correct_avg} | tokens/query: {tokens_avg} | tokens/correct: {tokens_correct_avg}")

    # print exa with only joins considered and synthetic table names predicted
    def analyze_special_cases(self):
        # eval file needs to be created first
        if os.path.exists(self.eval_path):
            with open(self.eval_path, "r") as f: 
                eval = json.load(f)
        else:
            raise Exception(f"Create eval file for {self.dataset}_{self.db_size} first.")
        
        # define level
        if self.f_suffix:
            level = "L2"
        else:
            level = "L1"
        
        synthetic_table_names = {} # dict of db_ids holding synthetic table names

        # monitoring values
        join_exa = 0
        join_duration = 0
        total_join_count = 0
        synthetic_exa = 0
        synthetic_total_count = 0

        for sample in eval:
            # analyze join samples
            if " join " in sample["sql_gold"].lower():
                join_exa += sample["execution_accuracy"]
                join_duration += sample["duration_seconds"]
                total_join_count += 1            
            
            # analyze synthetic tables used
            db_id = sample['db_id']
            if db_id not in synthetic_table_names: # synthetic tables of db_id not already extracted
                with open(f"{SCHEMAS_PATH}{self.dataset}/{db_id}.json", "r") as f:
                    original_schema = json.load(f)
                with open(f"{self.schemas_path}/{db_id}.json", "r") as f:
                    enlarged_schema = json.load(f)

                # add all tables of enlarged schemas that do not appear in original schema to synthetic names
                synthetic_table_names[db_id] = [tbl for tbl in enlarged_schema["schema"].keys() if tbl not in original_schema["schema"].keys()]
            
            pred_sql = sample["response"]["sql"] or ""

            # check if synthetic table reference exists
            synthetic_table_reference = False
            
            tables_used = extract_table_names(pred_sql)

            synthetic_table_reference = any(
                tbl.lower() in tables_used
                for tbl in synthetic_table_names[db_id]
            )

            # if synthetic table is referenced monitor
            if synthetic_table_reference:
                synthetic_exa += sample["execution_accuracy"]
                synthetic_total_count += 1
        
        join_exa_score = round(join_exa / total_join_count * 100, 2)
        duration_avg = round(join_duration / total_join_count, 2)
        if synthetic_total_count == 0:
            synthetic_exa_score = "-"
        else:
            synthetic_exa_score = round(synthetic_exa / synthetic_total_count * 100, 2)
        
        print(f"{self.dataset}-dev-{self.db_size} | {level} | {self.model} | Join Count: {total_join_count} | Join ExA: {join_exa_score} | Join Duration: {duration_avg} | Synthetic Count: {synthetic_total_count} | Synthetic ExA: {synthetic_exa_score}")


# find and extract table names from sql query
def extract_table_names(sql):
    sql = sql.lower()
    tables = set()

    for match in TABLE_REF_REGEX.finditer(sql):
        tbl1, tbl2 = match.groups()

        # if schema.table, tbl2 is the real table name
        if tbl2:
            tables.add(tbl2)
        else:
            tables.add(tbl1)

    return tables
