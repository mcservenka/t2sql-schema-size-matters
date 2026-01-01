#
# base on schema json representation
#

SQLITE_TYPEGROUP = {
    "INT": "INTEGER",
    "INTEGER": "INTEGER",
    "TEXT": "TEXT",
    "REAL": "REAL",
    "FLOAT": "REAL",
    "DOUBLE": "REAL",
    "DATE": "TEXT",
    "TIMESTAMP": "TEXT",
}

# retruns type and type group
def normalize_type(sql_type: str) -> tuple:
    t = sql_type.upper()
    if t in SQLITE_TYPEGROUP:
        return t, SQLITE_TYPEGROUP[t]
    # fallback
    return "TEXT", "TEXT"

def get_original_table_names(schema_original: dict) -> set:
    return set(schema_original.keys())

def get_original_column_names(schema_original: dict) -> set:
    cols = set()
    for t, obj in schema_original.items():
        for c in obj.get("columns", []):
            cols.add(c["name"])
    return cols

# column based on input attributes
def make_column(name: str, sql_type: str, pk: int = 0, notnull: bool = False) -> dict:
    t, tg = normalize_type(sql_type)
    return {
        "name": name,
        "type": t,
        "typegroup": tg,
        "notnull": bool(notnull),
        "pk": int(pk),
    }

# create CREATE TABLE statements
def sqlite_create_table_sql(table:str, obj:dict) -> str:
        cols = obj["columns"]
        pks = obj.get("primary_keys", [])
        fks = obj.get("foreign_keys", [])

        col_lines = []
        pk_inline = set(pks) if len(pks) == 1 else set()

        for c in cols:
            name = c["name"]
            ctype = c["type"]
            nn = " NOT NULL" if c.get("notnull", False) else ""
            pk = " PRIMARY KEY" if name in pk_inline else ""
            col_lines.append(f'"{name}" {ctype}{nn}{pk}')

        # composite PK
        if len(pks) > 1:
            pk_cols = ", ".join([f'"{k}"' for k in pks])
            col_lines.append(f"PRIMARY KEY ({pk_cols})")

        # foreign keys
        for fk in fks:
            # Note: your schema uses sourceTable/sourceColumn as the referenced table/col,
            # and targetColumn as the local column. (Names are a bit inverted but consistent.)
            ref_table = fk["sourceTable"]
            ref_col = fk["sourceColumn"]
            local_col = fk["targetColumn"]
            col_lines.append(
                f'FOREIGN KEY ("{local_col}") REFERENCES "{ref_table}"("{ref_col}")'
            )

        cols_sql = ",\n  ".join(col_lines)
        return f'CREATE TABLE "{table}" (\n  {cols_sql}\n);'