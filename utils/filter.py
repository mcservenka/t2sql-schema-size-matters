import re

# tokenizer (lowercase and snake/camel-case)
def normalize(text: str):
    text = text.lower()
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace("_", " ")
    return text.split()

# convert schema table to text
def table_to_text(table_name: str, table_def: dict):
    parts = [f"table {table_name}"]

    for col in table_def["columns"]:
        col_text = f"column {col['name']} type {col['typegroup']}"
        if col.get("pk", 0):
            col_text += " primary key"
        parts.append(col_text)

    for fk in table_def.get("foreign_keys", []):
        parts.append(
            f"foreign key {fk['sourceColumn']} references {fk['sourceTable']}"
        )

    return " ".join(parts)

