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
Initially please download the required ressources from nltk using:
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


