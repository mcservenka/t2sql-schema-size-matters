# Schema Size Matters: Context Pressure and Efficiency in Text-to-SQL

Text-to-SQL systems are commonly evaluated on benchmarks with small and simple database schemas, despite the fact that real-world databases structurally grow over time through the addition of tables and columns. As a result, the impact of schema size on model performance and cost remains largely unclear. This repository provides the source code for the paper **Schema Size Matters: Context Pressure and Efficiency in Text-to-SQL**, which addresses this gap by introducing a deterministic, structure-preserving schema scaling procedure that enlarges existing databases while keeping all original queries executable and semantically unchanged. Using scaled variants of Spider-dev with up to hundreds of tables, we evaluate two large language models under increasing schema-induced context pressure. Beyond execution accuracy, we introduce budget-normalized metrics capturing token consumption and runtime.
In order to reproduce the results, follow the instructions below.

>Cservenka, Markus. "Schema Size Matters: Context Pressure and Efficiency in Text-to-SQL", 2025.

Link to paper following soon...

## Ressources
To set up the environment, start by downloading the development sets of [Spider](https://yale-lily.github.io/spider) into `./data/datasets/spider/`. Then add the git submodule [Test-Suite-Evaluation](https://github.com/taoyds/test-suite-sql-eval) to  `./external/`. We will need these scripts later for computing the execution accuracy of the predicted SQLs. Make sure to define the OpenAI TogetherAI API keys in your environment variables as `OPENAI_API_KEY`, `OPENAI_API_ORGANIZATION`, `OPENAI_API_PROJECT` and `TOGETHERAI_API_KEY`. We further recommend using the `dotenv`-package.

## Environment Setup
Now set up the Python environment:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Next please download the required ressources from the `nltk` package using:
```
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Experiment
Follow the steps down below to recreate the experiment.

### Schema Representation
First we need to build the schema representation objects of the original Spider databases using `prepare_schema.py`. This will store each database schema of spider-dev (original) as a json-file in `data/schemas/`
```
python prepare_schema.py
```

### Candidate Name Selection
Before generating the enlarged schema variants, we first construct a pool of candidate names to be used for synthetic tables and columns. As outlined in the original paper, several configuration parameters can be adjusted to control both the size and the characteristics of this candidate set that is selected on predefined, noun-only vocabulary. In the following, you can find a list of the most influential parameters and their corresponding effects as well as the values we used in our experiment:
* **Similarity Minimum** `sim_min`: The minimum of similarity allowed between the schema centroid and a word to be considered domain-related (`sim_min=0.55`).
* **Similarity Maximum** `sim_max`: The maximum of similarity allowed between the schema centroid and a word to be considered domain-related but not too close to it's inherit word-space (`sim_max=0.25`).
* **Similarity Ambiguity** `sim_ambiguity`: The maximum of similarity allowed between a word to any of the original schema names (`sim_ambiguity=0.75`). This ensures that no selected word is synonymous to any schema name.
* **Candidates Minimum** `min_candidates`: The minimum number of candidate words that should be generated for a single database (`min_candidates=400`).
* **Candidates Maximum** `max_candidates`: The maximum number of candidate words that should be generated for a single database (`max_candidates=1000`).
* **Anchor k** `anchor_k`: The number of semantically related nouns used for stabilizing the embedding space in case of small schemas with a narrow semantic representation (`anchor_k=30`).

To generate the candidate words in `data/candidates` you can run:
```
python generate_names.py \
    --sim_min 0.55 \
    --sim_max 0.25 \
    --sim_ambiguity 0.75 \
    --min_candidates 400 \
    --max_candidates 1000 \
    --anchor_k 30
```
