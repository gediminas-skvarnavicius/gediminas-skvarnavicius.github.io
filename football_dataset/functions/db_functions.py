from duckdb import DuckDBPyConnection
from IPython.display import display, Markdown
from tabulate import tabulate
import pandas as pd


def check_db_nulls(con: DuckDBPyConnection, table: str, row_limit: int = 5) -> None:
    """
    Check for null values in a given SQLite table.

    Args:
        table (str): The name of the table to check for null values.
        con (DuckDBPyConnection): The DuckDB connection object.
        row_limit (int): The limit of rows with nulls to display.

    Returns:
        None

    Raises:
        ValueError: If the specified table does not exist in the query result.

    Examples:
        >>> connection = DuckDBPyConnection('database.db')
        >>> check_db_nulls('Player', connection)
        No nulls in column_name Column
        Nulls found in column_name Column: [<resultset.Result at 0x7f6a811e6b80>]
    """
    reserved_keywords = ["CROSS", "JOIN", "SELECT", "WHERE"]

    tables = (
        con.query(
            f"""--sql
SELECT table_name as 'tables',
FROM information_schema.columns
GROUP BY table_name
"""
        )
        .to_df()["tables"]
        .values
    )
    if table not in tables:
        raise ValueError(f"Table '{table}' does not exist in the query result.")

    table_columns = (
        con.query(
            f"""--sql
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table}'"""
        )
        .to_df()["column_name"]
        .values
    )

    for col in table_columns:
        if col.upper() in reserved_keywords:
            col = f'"{col}"'
        nulls = con.query(
            f"""--sql
                SELECT * FROM {table}
                WHERE {col} IS NULL
                """
        )
        if not nulls:
            display(Markdown(f"No nulls in {col} Column"))
        else:
            nulls = nulls.to_df()
            display(Markdown(f"{str(len(nulls))} null rows found in {col}:"))
            display(
                Markdown(
                    tabulate(
                        nulls[:row_limit],
                        showindex=False,
                        headers="keys",
                        tablefmt="pipe",
                    )
                )
            )


def check_db_refs(
    con: DuckDBPyConnection,
    tab1: str,
    tab2: str,
    key_name: str,
    key_name2: str = None,
    row_limit: int = 5,
) -> None:
    """
    Check if all entries in one table are referenced in another table based on a specified key.

    Args:
        tab1 (str): Name of the first table.
        tab2 (str): Name of the second table.
        key_name (str): Name of the key used for referencing.
        con (DuckDBPyConnection): The DuckDB connection object.
        row_limit (int): The limit of rows with nulls to display.

    Returns:
        None

    Raises:
        ValueError: if there are null values in specified key column in table 1.

    Examples:
        >>> connection = DuckDBPyConnection('database.db')
        >>> check_db_refs('Table1', 'Table2', 'id', connection)
        All entries in table 1 are referenced in table 2
        Nulls found in key_name Column: [<resultset.Result at 0x7f6a811e6b80>]
    """
    if not key_name2:
        key_name2 = key_name

    nulls = con.query(
        f"""--sql
        SELECT DISTINCT p.{key_name}
        FROM {tab1} p
        LEFT JOIN {tab2} pa ON p.{key_name} = pa.{key_name2}
        WHERE pa.{key_name2} IS NULL
        """
    )
    if not nulls:
        display(Markdown(f"All entries in {tab1} are referenced in {tab2}."))
    else:
        vals = con.query(
            f"""--sql
        SELECT DISTINCT p.{key_name}
        FROM {tab1} p
        LEFT JOIN {tab2} pa ON p.{key_name} = pa.{key_name2}
        WHERE pa.{key_name2} IS NULL
        """
        ).to_df()
        if vals.isna().any().any():
            raise ValueError(f"Key {key_name} is missing from some entries in {tab1}.")

        missing_entries = con.query(
            f"""--sql
          SELECT *
          FROM {tab1}
          WHERE {key_name} IN ({", ".join(vals[key_name].values.tolist())})
          """
        ).to_df()

        display(
            Markdown(
                f"{str(len(vals))} {key_name} values from {tab1} were found to be unreferenced in {tab2}:"
            )
        )
        display(
            Markdown(
                tabulate(
                    missing_entries[:row_limit],
                    showindex=False,
                    headers="keys",
                    tablefmt="pipe",
                )
            )
        )
