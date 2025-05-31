"""
PostgresSearcher is a class that searches for items in a PostgreSQL
database using a hybrid search strategy.
"""

# pylint:disable=import-error,missing-function-docstring,missing-class-docstring,unsupported-binary-operation
from typing import Union
from sqlalchemy import Float, Integer, column, select, text

from app.rag.embedding import Embedding
from app.db.database import get_db_session

embedding_util = Embedding()


class PostgresSearcher:

    embed_model: str = "text-embedding-3-small"

    def __init__(
        self,
        db_model,
        embed_dimensions: Union[int, None] = 1536,
        join_tables: Union[dict, None] = None,
    ):
        self.db_model = db_model
        self.embed_dimensions = embed_dimensions
        self.join_tables = join_tables or {}

    def build_filter_clause(self, filters) -> tuple[str, str]:
        """
        Builds SQL filter clauses from a list of filter dictionaries.
        Handles various operators including JSON operators and array operations.
        Returns a tuple of (WHERE clause, AND clause).
        """
        if not filters:
            return "", ""

        def format_value(value, operator):
            if isinstance(value, str):
                # Handle JSON operators
                if operator in [">>", "->"]:
                    return f"'{value}'"
                return f"'{value}'"
            elif isinstance(value, list):
                if operator == "&&":  # Array overlap operator
                    # Convert values to uppercase and replace spaces with underscores for enum compatibility
                    _vals = [f'"{v}"' for v in value]
                    return "'{" + ",".join(_vals) + "}'"
                elif operator == "@>":  # Array contains operator
                    # Convert values to uppercase and replace spaces with underscores for enum compatibility
                    _vals = [v for v in value]
                    return f"ARRAY[{','.join(repr(v) for v in _vals)}]"
                else:  # IN operator
                    return (
                        "("
                        + ",".join(
                            [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
                        )
                        + ")"
                    )
            elif value is None:
                return "NULL"
            return str(value)

        def build_clause(filter_dict):
            column = filter_dict["column"]
            operator = filter_dict["comparison_operator"]
            value = filter_dict["value"]
            value_type = filter_dict.get(
                "type", "string"
            )  # Default to string if not specified

            # Handle array enum types
            if value_type.endswith("[]"):
                enum_type = value_type[:-2]  # Remove the [] suffix
                if isinstance(value, list):
                    if operator == "&&":
                        _vals = [f'"{v}"' for v in value]
                        array_value = "'{" + ",".join(_vals) + "}'"
                        return f"CAST({column} AS {enum_type}[]) && {array_value}::{enum_type}[]"
                    elif operator == "@>":
                        _vals = [v for v in value]
                        array_value = f"ARRAY[{','.join(repr(v) for v in _vals)}]"
                        return f"CAST({column} AS {enum_type}[]) @> {array_value}::{enum_type}[]"

            # Handle JSON path operators
            if "->>" in column:
                column_parts = column.split("->>")
                base_column = column_parts[0].strip()
                json_path = column_parts[1].strip().strip("'")
                # Cast the result to numeric for numeric comparisons
                if isinstance(value, (int, float)):
                    # Use ->> for JSON text access and then CAST to numeric
                    # handle custom enum values case
                    return f"CAST({base_column}->>{format_value(json_path, '>>')} AS NUMERIC) {operator} {value}"
                # Use ->> for JSON text access for non-numeric comparisons
                return f"{base_column}->>{format_value(json_path, '>>')} {operator} {format_value(value, operator)}"

            # Handle NULL comparisons
            if value is None:
                return (
                    f"{column} IS NULL" if operator == "=" else f"{column} IS NOT NULL"
                )

            # Handle boolean values
            if isinstance(value, bool):
                return f"{column} {operator} {str(value)}"

            return f"{column} {operator} {format_value(value, operator)}"

        filter_clauses = [build_clause(filter) for filter in filters]
        filter_clause = " AND ".join(filter_clauses)

        if filter_clause:
            return f"WHERE {filter_clause}", f"AND {filter_clause}"
        return "", ""

    def build_join_clause(self, filters) -> str:
        """
        Build JOIN clauses based on filters that reference other tables.
        """
        if not self.join_tables:
            return ""

        # Check if any filters reference joined tables
        referenced_tables = set()
        for filter_dict in filters or []:
            column = filter_dict["column"]
            if "." in column:
                table_alias = column.split(".")[0]
                referenced_tables.add(table_alias)

        join_clauses = []
        for table_alias in referenced_tables:
            if table_alias in self.join_tables:
                join_info = self.join_tables[table_alias]
                join_clauses.append(
                    f"JOIN {join_info['table']} {table_alias} ON {join_info['on']}"
                )

        return " ".join(join_clauses)

    def search(
        self,
        query_text: Union[str, None],
        query_vector: Union[list[float], list],
        top: int = 5,
        filters: Union[list[dict], None] = None,
    ):
        filter_clause_where, filter_clause_and = self.build_filter_clause(filters)
        join_clause = self.build_join_clause(filters)

        table_name = self.db_model.__tablename__
        embedding_field_name = self.db_model.get_embedding_field()
        search_text_field_name = self.db_model.get_text_search_field()

        # Build the base table reference with alias
        base_table = f'"{table_name}" {table_name.replace("_", "")}'

        vector_query = f"""
            SELECT {table_name.replace("_", "")}.id, RANK () OVER (ORDER BY {table_name.replace("_", "")}.{embedding_field_name} <=> :embedding) AS rank
                FROM {base_table}
                {join_clause}
                {filter_clause_where}
                ORDER BY {table_name.replace("_", "")}.{embedding_field_name} <=> :embedding
                LIMIT 20
            """

        fulltext_query = f"""
            SELECT {table_name.replace("_", "")}.id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', {table_name.replace("_", "")}.{search_text_field_name}), plainto_tsquery('english', :query)) DESC)
                FROM {base_table}
                {join_clause}
                WHERE to_tsvector('english', {table_name.replace("_", "")}.{search_text_field_name}) @@ plainto_tsquery('english', :query) {filter_clause_and}
                ORDER BY ts_rank_cd(to_tsvector('english', {table_name.replace("_", "")}.{search_text_field_name}), plainto_tsquery('english', :query)) DESC
                LIMIT 20
            """

        hybrid_query = f"""
        WITH vector_search AS (
            {vector_query}
        ),
        fulltext_search AS (
            {fulltext_query}
        )
        SELECT
            COALESCE(vector_search.id, fulltext_search.id) AS id,
            COALESCE(1.0 / (:k + vector_search.rank), 0.0) +
            COALESCE(2.0 / (:k + fulltext_search.rank), 0.0) AS score
        FROM vector_search
        FULL OUTER JOIN fulltext_search ON vector_search.id = fulltext_search.id
        ORDER BY score DESC
        LIMIT 20
        """

        if query_text is not None and len(query_vector) > 0:
            sql = text(hybrid_query).columns(
                column("id", Integer), column("score", Float)
            )
        elif len(query_vector) > 0:
            sql = text(vector_query).columns(
                column("id", Integer), column("rank", Integer)
            )
        elif query_text is not None:
            sql = text(fulltext_query).columns(
                column("id", Integer), column("rank", Integer)
            )
        else:
            raise ValueError("Both query text and query vector are empty")

        results = []
        with get_db_session() as db_session:
            results = (
                db_session.execute(
                    sql,
                    {"embedding": str(query_vector), "query": query_text, "k": 60},
                )
            ).fetchall()

        # Convert results to models
        items = []
        for id, _ in results[:top]:
            with get_db_session() as db_session:
                item = db_session.execute(
                    select(self.db_model).where(self.db_model.id == id)
                )
                _item = item.scalar()
                if _item:
                    items.append(_item)
        return items

    def search_and_embed(
        self,
        query_text: Union[str, None] = None,
        top: int = 5,
        enable_vector_search: bool = True,
        enable_text_search: bool = True,
        filters: Union[list[dict], None] = None,
    ):
        """
        Search items by query text. Optionally converts the query text to a
        vector if enable_vector_search is True.
        """
        vector: list[float] = []
        if enable_vector_search and query_text is not None:
            vector = embedding_util.generate(
                query_text,
                self.embed_dimensions,
            )
        if not enable_text_search:
            query_text = None

        return self.search(query_text, vector, top, filters)
