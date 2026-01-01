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

To generate the candidate words in `data/candidates/` you can run:
```
python generate_names.py \
    --sim_min 0.55 \
    --sim_max 0.25 \
    --sim_ambiguity 0.75 \
    --min_candidates 400 \
    --max_candidates 1000 \
    --anchor_k 30
```

### Schema Scaling
Next we can start generating the augmented dataset variants of spider-dev. You can determine which specific variant you want to generate by setting `target_size` - which determines the number of tables inserted into the original database - and `apply_level_2` - which decides whether level 1 (default) or level 2 of schema inflation is performed. In level 1 the schema scaler performs pure schema inflation without introducing ambiguity relative to the original schema. In level 2 original-inspired tables and join-competition are introduced as well.
```
python enlarge_databases.py \
    --target_size 100 \
    --apply_level_2
```
The newly created datasets are stored in `data/datasets/` and level 2 variants are marked with an `f` suffix. The schema scaler further generates metadata-files - that provide information about table and foreign key counts before and after augmentation - and stores them in `data/metadata/`. The corresponding JSON-files containing the schema representation are stored in `data/schemas/`.

### Prompt Model
Now using our augmented versions of spider-dev, we can start prompting the models. In terms of LLMs utilized within this study, one open- (`gpt-5.2` via OpenAI API) and one closed-source (`llama-3.3-70B` via TogetherAI API) LLM were tested, which is common in this field of research. To generate the results in `data/reesults/` run the following command for each variant:
```
python prompt_model.py \
    --model "gpt-5.2" \
    --db_size 100 \
    --apply_level_2 \
    --schema_filter "bm25"
```
Make sure that the dataset variant you select really exists in `data/datasets/` and `data/schemas/`, respectively. The `schema_filter` parameter decides whether one of the implemented filtering methods (`bm25`, `dense`) is applied before prompting the model.

### Evaluation
Eventually, you can evaluate the responses by running `evaluate_results.py`. This will add the evaluation scores to your response objects and create a new file in `data/results/` and print the evaluation results to the console.
```
python evaluate_results.py \
    --model "gpt-5.2" \
    --db_size 100 \
    --apply_level_2 \
    --schema_filter "bm25"
```
Again make sure the results for the selected variants were generated beforehand.

## Experiment Results
Down below we illustrated the official results of our paper. Please note that - although our schema scaler behaves inherently deterministic - the results may vary after rerunning the experiment due to the inherent stochasticity of the LLM. For detailed evaluation results feel free to check out chapter 5 of the paper.

### Aggregate Analysis
GPT-5.2 Performance across Variants.
<table>
  <thead>
    <tr>
      <th>Level</th>
      <th>Variant</th>
      <th>ExA</th>
      <th>TE</th>
      <th>LE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Original</strong></td>
      <td>spider-dev-0</td>
      <td>76.79</td>
      <td>614</td>
      <td>1.21</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-50</td>
      <td>74.47</td>
      <td>3,898</td>
      <td>1.45</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-100</td>
      <td>74.08</td>
      <td>7,344</td>
      <td>1.42</td>
    </tr>

    <tr>
      <td><strong>Level 1</strong></td>
      <td>spider-dev-250</td>
      <td>75.15</td>
      <td>17,759</td>
      <td>1.57</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-500</td>
      <td>74.47</td>
      <td>35,527</td>
      <td>1.88</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-800</td>
      <td>73.98</td>
      <td>57,135</td>
      <td>2.28</td>
    </tr>

    <tr>
      <td><strong>Level 2</strong></td>
      <td>spider-dev-50</td>
      <td>73.60</td>
      <td>3,818</td>
      <td>1.55</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-100</td>
      <td>73.79</td>
      <td>7,016</td>
      <td>1.69</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-250</td>
      <td>74.76</td>
      <td>16,834</td>
      <td>1.88</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-500</td>
      <td>73.89</td>
      <td>33,454</td>
      <td>1.93</td>
    </tr>
    <tr>
      <td></td>
      <td>spider-dev-800</td>
      <td>73.21</td>
      <td>53,317</td>
      <td>2.21</td>
    </tr>
  </tbody>
</table>
