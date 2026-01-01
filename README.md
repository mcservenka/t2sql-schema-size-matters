# Schema Size Matters: Context Pressure and Efficiency in Text-to-SQL

Text-to-SQL systems are commonly evaluated on benchmarks with small and simple database schemas, despite the fact that real-world databases structurally grow over time through the addition of tables and columns. As a result, the impact of schema size on model performance and cost remains largely unclear. This repository provides the source code for the paper **Schema Size Matters: Context Pressure and Efficiency in Text-to-SQL**, which addresses this gap by introducing a deterministic, structure-preserving schema scaling procedure that enlarges existing databases while keeping all original queries executable and semantically unchanged. Using scaled variants of Spider-dev with up to hundreds of tables, we evaluate two large language models under increasing schema-induced context pressure. Beyond execution accuracy, we introduce budget-normalized metrics capturing token consumption and runtime.
In order to reproduce the results, follow the instructions below.

>Cservenka, Markus. "Schema Size Matters: Context Pressure and Efficiency in Text-to-SQL", 2025.

Link to paper following soon...



