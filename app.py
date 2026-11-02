
import re
import hashlib
from io import BytesIO

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from openai import OpenAI


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Data Analyst")
st.caption("Upload a CSV or Excel file.")
st.info("Please select the column names in your dataset for accurate analysis.")

MAX_FILE_MB = 200


# =========================================================
# NVIDIA NEMOTRON CONNECTION
# =========================================================


# =========================================================
# NVIDIA NEMOTRON CONNECTION
# =========================================================

@st.cache_resource
def get_nvidia_client():
    api_key = st.secrets.get("NVIDIA_API_KEY")

    if not api_key:
        return None

    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=30,
    )


def ask_nemotron(question, analysis_context):
    """Explain calculated analysis results using NVIDIA Nemotron."""

    client = get_nvidia_client()

    if client is None:
        return "NVIDIA_API_KEY is missing from Streamlit Secrets."

    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert business data analyst. "
                        "Explain the supplied calculated results clearly. "
                        "Never invent data, figures, columns, or trends. "
                        "Use only the supplied results. "
                        "Keep your answer to 3 to 5 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Calculated results:\n{analysis_context}\n\n"
                        "Explain these results directly."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=500,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )

        message = response.choices[0].message

        if message.content and message.content.strip():
            return message.content.strip()

        return (
            "Nemotron returned no explanation text. "
            f"Finish reason: {response.choices[0].finish_reason}. "
            f"Message object: {message!r}"
        )

    except Exception as e:
        return f"NVIDIA API error: {str(e)}"


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = BytesIO(file_bytes)
    lower = filename.lower()

    if lower.endswith(".csv"):
        last_error = None

        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding=encoding)
            except UnicodeDecodeError as exc:
                last_error = exc

        raise ValueError(f"Could not decode this CSV file: {last_error}")

    if lower.endswith((".xlsx", ".xls")):
        buffer.seek(0)
        return pd.read_excel(buffer)

    raise ValueError("Please upload a CSV or Excel file.")


# =========================================================
# DATA TYPE DETECTION
# =========================================================

def parse_dates(series):
    """Detect dates stored as text."""

    if len(series.dropna()) == 0:
        return None

    try:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce")

    valid_ratio = (
        parsed.notna().sum()
        / max(series.notna().sum(), 1)
    )

    if valid_ratio >= 0.75:
        return parsed

    return None


def clean_numeric_series(series):
    """
    Convert common text-formatted numbers into numeric values.
    Handles commas, currency symbols, and percentage symbols.
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype("string").str.strip()

    # Accounting negatives: (123.45) becomes -123.45.
    cleaned = cleaned.str.replace(
        r"^\((.*)\)$",
        r"-\1",
        regex=True,
    )

    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace("₹", "", regex=False)
    cleaned = cleaned.str.replace("€", "", regex=False)
    cleaned = cleaned.str.replace("£", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)

    cleaned = cleaned.replace(
        ["", "NA", "N/A", "na", "n/a", "None", "none", "null", "NULL", "-"],
        pd.NA,
    )

    return pd.to_numeric(cleaned, errors="coerce")


def get_schema(df):
    """
    Return numeric columns, categorical columns, date columns,
    and parsed date values.
    """

    numeric = []
    date_map = {}

    # Detect date columns first.
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_map[col] = pd.to_datetime(
                df[col],
                errors="coerce",
            )

        elif any(
            token in str(col).lower()
            for token in ("date", "time", "timestamp")
        ):
            parsed = parse_dates(df[col])

            if parsed is not None:
                date_map[col] = parsed

    dates = list(date_map.keys())

    # Detect numeric columns, including numeric text columns.
    for col in df.columns:

        if col in dates:
            continue

        if pd.api.types.is_bool_dtype(df[col]):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
            continue

        values = df[col].dropna()

        if values.empty:
            continue

        converted = clean_numeric_series(values)
        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.90:
            name = str(col).lower()

            identifier_tokens = (
                "id",
                "postal",
                "zip",
                "postcode",
                "phone",
                "telephone",
                "code",
            )

            looks_like_identifier = any(
                token in name.split()
                or name.endswith("_" + token)
                or name.endswith(token)
                for token in identifier_tokens
            )

            if not looks_like_identifier:
                numeric.append(col)

    categorical = [
        col for col in df.columns
        if col not in numeric and col not in dates
    ]

    return numeric, categorical, dates, date_map


# =========================================================
# EXCEL EXPORT
# =========================================================

def dataframe_to_excel_bytes(
    dataframe: pd.DataFrame,
    sheet_name: str = "Analysis",
) -> bytes:

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(
            writer,
            index=False,
            sheet_name=sheet_name[:31] or "Analysis",
        )

    return output.getvalue()


# =========================================================
# QUESTION INTERPRETATION
# =========================================================

def norm(text):
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(text).lower(),
    ).strip()


def mentioned_columns(question, columns):
    """Find column names explicitly mentioned in a question."""

    q = norm(question)
    hits = []

    for col in columns:
        normalized = norm(col)

        if normalized and re.search(
            r"(?<!\w)" + re.escape(normalized) + r"(?!\w)",
            q,
        ):
            hits.append(col)

    return sorted(
        hits,
        key=lambda col: len(norm(col)),
        reverse=True,
    )


def detect_missing_requested_column(question, df):
    """
    Detect when a question requests a measure or field
    that is not present in the uploaded dataset.
    """

    q = norm(question)

    aliases = {
        "profit": ["profit", "profits", "earnings"],
        "revenue": ["revenue", "turnover"],
        "sales": ["sales"],
        "quantity": ["quantity", "units sold"],
        "discount": ["discount", "discounts"],
        "cost": ["cost", "costs", "expenses", "expense"],
        "salary": ["salary", "salaries", "wages"],
        "age": ["age"],
        "price": ["price", "prices"],
        "rating": ["rating", "ratings", "score"],
    }

    existing_columns = {
        norm(col): col for col in df.columns
    }

    existing_names = set(existing_columns)

    for canonical, terms in aliases.items():
        mentioned = any(
            re.search(
                r"(?<!\w)" + re.escape(norm(term)) + r"(?!\w)",
                q,
            )
            for term in terms
        )

        if not mentioned:
            continue

        matching_column_exists = any(
            norm(term) in existing_names
            or any(
                norm(term) in column_name.split()
                for column_name in existing_names
            )
            for term in terms
        )

        if not matching_column_exists:
            return canonical

    return None


def choose_measure(question, numeric, explicit):
    """Select the numeric measure requested by the user."""

    for col in explicit:
        if col in numeric:
            return col

    q = norm(question)

    hints = [
        (
            ("sales", "revenue", "turnover", "income"),
            ("sales", "revenue", "turnover", "amount"),
        ),
        (
            ("profit", "earnings", "margin"),
            ("profit", "earnings", "margin"),
        ),
        (
            ("quantity", "units", "items sold", "volume"),
            ("quantity", "units", "volume"),
        ),
        (
            ("discount",),
            ("discount",),
        ),
        (
            ("cost", "expense", "spend"),
            ("cost", "expense", "spend"),
        ),
        (
            ("price",),
            ("price",),
        ),
        (
            ("salary", "salaries", "wage", "pay"),
            ("salary", "salaries", "wage", "pay"),
        ),
        (
            ("age",),
            ("age",),
        ),
        (
            ("score", "rating"),
            ("score", "rating"),
        ),
    ]

    for triggers, tokens in hints:
        if any(trigger in q for trigger in triggers):
            for col in numeric:
                name = norm(col)

                if any(token in name.split() for token in tokens):
                    return col

    if len(numeric) == 1:
        return numeric[0]

    return None


def choose_category(question, categorical, explicit, numeric=None):
    """Select the category used for grouping or comparison."""

    for col in explicit:
        if col in categorical:
            return col

    q = norm(question)

    hints = [
        (
            ("product", "products", "item", "items", "sku"),
            ("product", "item", "sku"),
        ),
        (
            ("customer", "customers", "client", "clients"),
            ("customer", "client", "buyer"),
        ),
        (
            ("region", "regional", "area", "territory"),
            ("region", "area", "territory", "state", "country"),
        ),
        (
            ("ship mode", "shipping", "shipment"),
            ("ship mode", "ship", "shipping", "shipment"),
        ),
        (
            ("category", "categories"),
            ("category",),
        ),
        (
            ("segment", "segments"),
            ("segment",),
        ),
        (
            ("department", "departments"),
            ("department", "dept"),
        ),
        (
            ("city", "cities"),
            ("city",),
        ),
        (
            ("state", "states"),
            ("state",),
        ),
        (
            ("country", "countries"),
            ("country",),
        ),
    ]

    numeric = numeric or []

    groupable = list(categorical) + [
        col for col in numeric
        if any(
            token in norm(col)
            for token in ("id", "code", "number", "postal", "zip")
        )
    ]

    for triggers, tokens in hints:
        if any(trigger in q for trigger in triggers):
            for col in groupable:
                name = norm(col)

                if any(token in name.split() for token in tokens):
                    return col

    if len(categorical) == 1:
        return categorical[0]

    return None


# =========================================================
# ANALYSIS ENGINE
# =========================================================

def answer_question(df, question):

    q = question.lower().strip()

    numeric, categorical, dates, date_map = get_schema(df)

    explicit = mentioned_columns(
        question,
        list(df.columns),
    )

    missing_column = detect_missing_requested_column(
        question,
        df,
    )

    if missing_column:
        return {
            "answer": (
                f"⚠️ The requested column **'{missing_column}'** "
                "was not found in your dataset.\n\n"
                f"**Available columns:** "
                f"{', '.join(map(str, df.columns))}\n\n"
                "Please check the column name or upload a dataset "
                "that contains this information."
            ),
            "table": None,
            "figure": None,
        }

    measure = choose_measure(
        question,
        numeric,
        explicit,
    )

    category = choose_category(
        question,
        categorical,
        explicit,
        numeric,
    )

    if category is None:
        category = next(
            (
                col for col in explicit
                if col != measure and col not in dates
            ),
            None,
        )

    date_col = next(
        (col for col in explicit if col in dates),
        None,
    )

    if date_col is None and len(dates) == 1:
        date_col = dates[0]

    def result(answer, table=None, fig=None):
        return {
            "answer": answer,
            "table": table,
            "figure": fig,
        }

    def measure_values(col):
        return clean_numeric_series(df[col])

    # -----------------------------------------------------
    # HELP
    # -----------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "help",
            "what can you do",
            "example questions",
        )
    ):
        return result(
            "### Try questions like these\n"
            "- What is the total Sales?\n"
            "- Which are the top 5 products by Sales?\n"
            "- Compare Sales by Ship Mode.\n"
            "- Show monthly Sales trends.\n"
            "- Which columns have missing values?\n"
            "- Show correlations between numeric columns.\n"
            "- Summarize this dataset."
        )

    # -----------------------------------------------------
    # DATASET OVERVIEW
    # -----------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "overview",
            "summarize dataset",
            "summary of dataset",
            "dataset summary",
            "data quality",
            "missing values",
            "duplicates",
            "how many rows",
            "how many columns",
            "complete summary",
            "summarize this dataset",
        )
    ):
        missing = df.isna().sum()

        table = pd.DataFrame({
            "Column": [str(col) for col in df.columns],
            "Data type": [str(df[col].dtype) for col in df.columns],
            "Missing values": [
                int(missing[col]) for col in df.columns
            ],
            "Unique values": [
                int(df[col].nunique(dropna=True))
                for col in df.columns
            ],
        })

        return result(
            f"### Dataset overview\n"
            f"- **Rows:** {len(df):,}\n"
            f"- **Columns:** {len(df.columns):,}\n"
            f"- **Missing cells:** {int(missing.sum()):,}\n"
            f"- **Duplicate rows:** {int(df.duplicated().sum()):,}",
            table,
        )

    # -----------------------------------------------------
    # CORRELATIONS
    # -----------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "correlation",
            "correlations",
            "correlate",
            "relationship between",
        )
    ):
        if len(numeric) < 2:
            return result(
                "Correlation analysis needs at least two numeric columns."
            )

        numeric_df = pd.DataFrame({
            col: measure_values(col)
            for col in numeric
        })

        corr = numeric_df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation matrix",
        )

        return result(
            "### Correlation matrix\n"
            "Values range from -1 to +1. Correlation does not prove causation.",
            corr,
            fig,
        )

    # -----------------------------------------------------
    # TRENDS OVER TIME
    # -----------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "trend",
            "over time",
            "monthly",
            "daily",
            "weekly",
            "yearly",
            "by month",
            "by year",
            "quarterly",
            "changed over time",
        )
    ):

        if measure is None or date_col is None:
            return result(
                "To make a trend chart, I need a numeric measure "
                "and a date column.\n\n"
                f"**Numeric columns:** {', '.join(map(str, numeric)) or 'none'}\n\n"
                f"**Date columns detected:** {', '.join(map(str, dates)) or 'none'}\n\n"
                "Try naming both, for example: "
                "'Show monthly Sales by Order Date'."
            )

        temp = pd.DataFrame({
            "date": date_map[date_col],
            "value": measure_values(measure),
        }).dropna()

        if temp.empty:
            return result(
                f"No valid dates and numeric values were found "
                f"for {date_col} and {measure}."
            )

        if "daily" in q or "by day" in q:
            freq = "D"
        elif "weekly" in q or "by week" in q:
            freq = "W"
        elif "yearly" in q or "by year" in q:
            freq = "YE"
        elif "quarter" in q:
            freq = "QE"
        else:
            freq = "ME"

        temp = (
            temp.set_index("date")["value"]
            .resample(freq)
            .sum()
            .rename(measure)
            .reset_index()
        )

        fig = px.line(
            temp,
            x="date",
            y=measure,
            markers=True,
            title=f"{measure} over time",
        )

        return result(
            f"### Trend: {measure} over time\n"
            f"Grouped by the requested time frequency using sum aggregation.",
            temp,
            fig,
        )

    # -----------------------------------------------------
    # RANKING AND TOP/BOTTOM QUESTIONS
    # -----------------------------------------------------

    rank_match = re.search(
        r"\b(top|bottom)\s+(\d+)\b",
        q,
    )

    is_rank = bool(rank_match) or any(
        phrase in q
        for phrase in (
            "highest",
            "lowest",
            "rank",
            "best",
            "worst",
            "most sales",
            "least sales",
        )
    )

    count_intent = any(
        phrase in q
        for phrase in (
            "number of",
            "count of",
            "count the",
            "how many",
        )
    )

    # Count records by group.
    if (
        count_intent
        and category is not None
        and any(word in q for word in ("each", "per", "by", "every"))
    ):
        table = (
            df.groupby(category, dropna=False)
            .size()
            .rename("Count")
            .reset_index()
            .sort_values(
                "Count",
                ascending=False,
                na_position="last",
            )
        )

        fig = px.bar(
            table.head(40),
            x=category,
            y="Count",
            title=f"Record count by {category}",
        )

        return result(
            f"### Record count by {category}",
            table,
            fig,
        )

    # Rank individual records.
    row_level_intent = any(
        phrase in q
        for phrase in (
            "who has",
            "which employee",
            "employees by",
            "employee by",
            "people by",
            "person by",
        )
    )

    if measure is not None and (
        row_level_intent
        or (is_rank and category is None)
    ):
        descending = not any(
            phrase in q
            for phrase in (
                "bottom",
                "lowest",
                "worst",
                "least",
            )
        )

        temp = df.copy()
        temp["__analysis_value__"] = measure_values(measure)

        temp = temp.sort_values(
            "__analysis_value__",
            ascending=not descending,
            na_position="last",
        )

        limit = (
            min(int(rank_match.group(2)), 100)
            if rank_match
            else 1
            if any(word in q for word in ("who has", "which employee"))
            else 10
        )

        table = temp.head(limit).drop(
            columns=["__analysis_value__"]
        )

        label = (
            "Highest"
            if descending and limit == 1
            else "Lowest"
            if not descending and limit == 1
            else "Top"
            if descending
            else "Bottom"
        )

        fig = px.bar(
            table,
            x=measure,
            y=table.index.astype(str),
            orientation="h",
            title=f"{label} {len(table)} records by {measure}",
        )

        return result(
            f"### {label} {len(table)} records by {measure}",
            table,
            fig,
        )

    # Rank categories by record count.
    if is_rank and count_intent and category is not None:
        table = (
            df.groupby(category, dropna=False)
            .size()
            .rename("Count")
            .reset_index()
        )

        descending = not any(
            word in q
            for word in ("bottom", "lowest", "worst", "fewest", "least")
        )

        table = table.sort_values(
            "Count",
            ascending=not descending,
            na_position="last",
        )

        limit = (
            min(int(rank_match.group(2)), 100)
            if rank_match
            else 10
        )

        table = table.head(limit)

        label = "Top" if descending else "Bottom"

        fig = px.bar(
            table,
            x=category,
            y="Count",
            title=f"{label} {len(table)} {category} by count",
        )

        return result(
            f"### {label} {len(table)} {category} by count",
            table,
            fig,
        )

    # Rank categories by numeric measure.
    if is_rank:
        if measure is None or category is None:
            return result(
                "To rank items, I need a numeric measure and a category column.\n\n"
                f"**Numeric columns:** {', '.join(map(str, numeric)) or 'none'}\n\n"
                f"**Category columns:** {', '.join(map(str, categorical)) or 'none'}\n\n"
                "Example: 'Which are the top 5 products by Sales?'"
            )

        agg_name = (
            "mean"
            if any(word in q for word in ("average", "mean", "avg"))
            else "sum"
        )

        temp = pd.DataFrame({
            category: df[category],
            measure: measure_values(measure),
        })

        table = (
            temp.groupby(category, dropna=False)[measure]
            .agg(agg_name)
            .reset_index()
        )

        descending = not any(
            word in q
            for word in ("bottom", "lowest", "worst", "least")
        )

        table = table.sort_values(
            measure,
            ascending=not descending,
            na_position="last",
        )

        limit = (
            min(int(rank_match.group(2)), 100)
            if rank_match
            else 1
            if any(word in q for word in ("highest", "lowest", "best", "worst"))
            else 10
        )

        table = table.head(limit)

        label = (
            "Highest"
            if descending and limit == 1
            else "Lowest"
            if not descending and limit == 1
            else "Top"
            if descending
            else "Bottom"
        )

        fig = px.bar(
            table,
            x=category,
            y=measure,
            title=f"{label} {len(table)} {category} by {agg_name} {measure}",
        )

        return result(
            f"### {label} {len(table)} {category} by {agg_name} {measure}\n"
            f"Values use {agg_name} aggregation by category.",
            table,
            fig,
        )

    # -----------------------------------------------------
    # CATEGORY COMPARISONS
    # -----------------------------------------------------

    if any(
        phrase in q
        for phrase in (
            "compare",
            "comparison",
            "breakdown",
            "by category",
            "by region",
            "by product",
            "across",
        )
    ):
        if measure is None or category is None:
            return result(
                "To compare groups, I need a numeric measure and a category column.\n\n"
                f"**Numeric columns:** {', '.join(map(str, numeric)) or 'none'}\n\n"
                f"**Category columns:** {', '.join(map(str, categorical)) or 'none'}"
            )

        agg_name = (
            "mean"
            if any(word in q for word in ("average", "mean", "avg"))
            else "sum"
        )

        temp = pd.DataFrame({
            category: df[category],
            measure: measure_values(measure),
        })

        table = (
            temp.groupby(category, dropna=False)[measure]
            .agg(agg_name)
            .reset_index()
            .sort_values(
                measure,
                ascending=False,
                na_position="last",
            )
        )

        fig = px.bar(
            table.head(40),
            x=category,
            y=measure,
            title=f"{agg_name.title()} {measure} by {category}",
        )

        return result(
            f"### {agg_name.title()} {measure} by {category}",
            table,
            fig,
        )

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    if any(
        word in q
        for word in ("missing", "null", "empty cells")
    ):
        table = (
            df.isna()
            .sum()
            .rename("Missing values")
            .to_frame()
        )

        table = table[
            table["Missing values"] > 0
        ].sort_values(
            "Missing values",
            ascending=False,
        )

        if table.empty:
            return result("### Missing values\nNo missing values found.")

        return result("### Missing values by column", table)

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    if any(word in q for word in ("duplicate", "duplicates")):
        return result(
            f"### Duplicate rows\n"
            f"Found **{int(df.duplicated().sum()):,}** duplicate row(s). "
            "No rows have been removed."
        )

    # -----------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------

    if measure is not None and any(
        word in q
        for word in (
            "total",
            "sum",
            "average",
            "mean",
            "median",
            "minimum",
            "maximum",
            "max",
            "min",
            "standard deviation",
            "describe",
            "summary",
            "how much",
        )
    ):
        s = measure_values(measure).dropna()

        if s.empty:
            return result(
                f"No valid numeric values were found in {measure}."
            )

        stats = pd.DataFrame({
            "Statistic": [
                "Count",
                "Sum",
                "Mean",
                "Median",
                "Minimum",
                "Maximum",
                "Standard deviation",
            ],
            "Value": [
                s.count(),
                s.sum(),
                s.mean(),
                s.median(),
                s.min(),
                s.max(),
                s.std(),
            ],
        })

        if "average" in q or "mean" in q:
            answer = f"### Average {measure}: **{s.mean():,.4g}**"

        elif "median" in q:
            answer = f"### Median {measure}: **{s.median():,.4g}**"

        elif "minimum" in q or re.search(r"\bmin\b", q):
            answer = f"### Minimum {measure}: **{s.min():,.4g}**"

        elif "maximum" in q or re.search(r"\bmax\b", q):
            answer = f"### Maximum {measure}: **{s.max():,.4g}**"

        elif "total" in q or "sum" in q or "how much" in q:
            answer = f"### Total {measure}: **{s.sum():,.4g}**"

        else:
            answer = f"### Summary statistics for {measure}"

        return result(answer, stats)

    # -----------------------------------------------------
    # DISTRIBUTIONS
    # -----------------------------------------------------

    if any(
        word in q
        for word in ("distribution", "histogram", "spread")
    ):
        if measure is None:
            return result(
                "Please name the numeric column whose distribution "
                "you want to inspect."
            )

        values = measure_values(measure)

        fig = px.histogram(
            x=values,
            title=f"Distribution of {measure}",
            labels={"x": measure},
        )

        return result(
            f"### Distribution of {measure}",
            values.describe().to_frame(),
            fig,
        )

    # -----------------------------------------------------
    # GENERAL FALLBACK
    # -----------------------------------------------------

    if any(
        word in q
        for word in ("total", "sum", "average", "mean", "median")
    ) and measure is None:
        return result(
            "I couldn't identify a numeric measure for your question.\n\n"
            f"**Available numeric columns:** "
            f"{', '.join(map(str, numeric)) or 'none'}\n\n"
            "Try using a column name shown in the Column profile."
        )

    return result(
        "I couldn't confidently identify the analysis you want.\n\n"
        "Try **help** for examples, or name the measure and category "
        "explicitly, such as 'top 5 products by Sales'.\n\n"
        "This free-hosting version uses a rule-based question interpreter "
        "rather than a large language model, so it supports common "
        "analysis requests but not every possible conversational question."
    )




# =========================================================
# AUTOMATED BUSINESS PERFORMANCE REPORT
# =========================================================

def detect_business_dimensions(df, categorical):
    """Detect likely geography, product, and customer dimensions by column name."""
    patterns = {
        "Geography": ("region", "state", "city", "country", "territory", "province", "market", "postal"),
        "Product": ("product", "item", "sku", "category", "subcategory", "brand"),
        "Customer": ("customer", "client", "buyer", "account", "consumer"),
    }
    found = {}
    for label, tokens in patterns.items():
        matches = [
            col for col in categorical
            if any(token in norm(col).split() for token in tokens)
            or any(norm(col).startswith(token + " ") for token in tokens)
        ]
        if matches:
            found[label] = matches
    return found


def detect_business_measures(df, numeric):
    """Prioritise common business measures; retain other numeric measures too."""
    priority = {
        "Sales / Revenue": ("sales", "revenue", "turnover", "amount", "gmv"),
        "Profit": ("profit", "earnings", "margin"),
        "Quantity": ("quantity", "qty", "units", "volume"),
        "Orders": ("order", "orders"),
        "Discount": ("discount",),
        "Cost": ("cost", "expense", "spend"),
        "Price": ("price",),
    }
    selected = {}
    used = set()
    for label, tokens in priority.items():
        for col in numeric:
            name = norm(col).split()
            if col not in used and any(token in name for token in tokens):
                selected[label] = col
                used.add(col)
                break
    for col in numeric:
        if col not in used:
            selected[str(col)] = col
    return selected


def _safe_numeric(df, col):
    return clean_numeric_series(df[col]).replace([np.inf, -np.inf], np.nan)


def _fmt(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def _period_comparison(df, date_values, measure, dimension=None, periods=2):
    """Compare the two latest complete calendar months available in the data."""
    temp = pd.DataFrame({
        "__date": pd.to_datetime(date_values, errors="coerce"),
        "__value": _safe_numeric(df, measure),
    })
    if dimension is not None:
        temp["__group"] = df[dimension].astype("string").fillna("(Missing)")
    temp = temp.dropna(subset=["__date", "__value"])
    if temp.empty:
        return None, None

    # Exclude the newest calendar month if the dataset does not reach month-end.
    max_date = temp["__date"].max()
    month_end = max_date + pd.offsets.MonthEnd(0)
    complete_through = max_date.normalize() >= month_end.normalize()
    latest_month = max_date.to_period("M")
    if not complete_through:
        latest_month -= 1

    available = sorted(temp["__date"].dt.to_period("M").unique())
    eligible = [p for p in available if p <= latest_month]
    if len(eligible) < 2:
        return None, None

    current_period, previous_period = eligible[-1], eligible[-2]
    current = temp[temp["__date"].dt.to_period("M") == current_period]
    previous = temp[temp["__date"].dt.to_period("M") == previous_period]
    current_total, previous_total = current["__value"].sum(), previous["__value"].sum()
    delta = current_total - previous_total
    pct = (delta / abs(previous_total) * 100) if previous_total != 0 else np.nan

    comparison = {
        "Metric": str(measure),
        "Previous month": str(previous_period),
        "Previous value": previous_total,
        "Latest complete month": str(current_period),
        "Latest value": current_total,
        "Change": delta,
        "Change %": pct,
        "Rows previous": len(previous),
        "Rows latest": len(current),
    }

    drivers = None
    if dimension is not None:
        old = previous.groupby("__group", dropna=False)["__value"].sum()
        new = current.groupby("__group", dropna=False)["__value"].sum()
        driver = pd.concat([old.rename("Previous"), new.rename("Latest")], axis=1).fillna(0)
        driver["Change"] = driver["Latest"] - driver["Previous"]
        if delta != 0:
            driver["Contribution to net change %"] = driver["Change"] / delta * 100
        else:
            driver["Contribution to net change %"] = np.nan
        drivers = (
            driver.sort_values("Change", ascending=(delta < 0))
            .reset_index(names=dimension)
        )
    return comparison, drivers


def build_business_report(df):
    """Calculate report tables and factual findings with Pandas only."""
    numeric, categorical, dates, date_map = get_schema(df)
    measures = detect_business_measures(df, numeric)
    dimensions = detect_business_dimensions(df, categorical)
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "measures": measures,
        "dimensions": dimensions,
        "date_columns": dates,
        "tables": {},
        "charts": {},
        "facts": [],
        "limitations": [],
    }

    # Overall performance: only detected numeric measures are included.
    overall_rows = []
    for label, col in measures.items():
        values = _safe_numeric(df, col).dropna()
        if values.empty:
            continue
        overall_rows.append({
            "Metric": label,
            "Column": str(col),
            "Valid records": int(values.count()),
            "Total": values.sum(),
            "Average": values.mean(),
            "Median": values.median(),
            "Minimum": values.min(),
            "Maximum": values.max(),
        })
    report["tables"]["Overall performance"] = pd.DataFrame(overall_rows)

    # Dimension summaries.
    for dimension_type, cols in dimensions.items():
        for col in cols:
            for label, measure in measures.items():
                temp = pd.DataFrame({
                    str(col): df[col].astype("string").fillna("(Missing)"),
                    str(measure): _safe_numeric(df, measure),
                }).dropna(subset=[str(measure)])
                if temp.empty:
                    continue
                grouped = (
                    temp.groupby(str(col), dropna=False)[str(measure)]
                    .agg(["sum", "mean", "count"])
                    .reset_index()
                    .sort_values("sum", ascending=False)
                )
                grouped.columns = [str(col), "Total", "Average", "Records"]
                safe_key = f"{dimension_type} - {col} - {measure}"
                report["tables"][safe_key] = grouped.head(100)
                if len(grouped):
                    report["facts"].append(
                        f"{dimension_type} field '{col}': highest {measure} group is "
                        f"{grouped.iloc[0][str(col)]} with total {_fmt(grouped.iloc[0]['Total'])}; "
                        f"lowest is {grouped.iloc[-1][str(col)]} with total {_fmt(grouped.iloc[-1]['Total'])}."
                    )

    # Customer concentration, if customer identifiers are available.
    for col in dimensions.get("Customer", []):
        likely_value = next(
            (v for k, v in measures.items() if k in ("Sales / Revenue", "Profit") or
             any(t in norm(v).split() for t in ("sales", "revenue", "profit", "amount"))),
            None,
        )
        if likely_value:
            customer_totals = pd.DataFrame({
                str(col): df[col].astype("string").fillna("(Missing)"),
                "Value": _safe_numeric(df, likely_value),
            }).dropna(subset=["Value"]).groupby(str(col))["Value"].sum().sort_values(ascending=False)
            total = customer_totals.sum()
            top_share = customer_totals.head(10).sum() / total * 100 if total else np.nan
            report["tables"][f"Top customers - {col}"] = customer_totals.head(20).rename("Value").reset_index()
            report["facts"].append(
                f"Top 10 customers account for {_fmt(top_share)}% of {likely_value}."
                if pd.notna(top_share) else "Customer concentration could not be calculated because total value is zero."
            )

    # Time series and latest complete month-over-month comparisons.
    if dates:
        date_col = dates[0]
        date_values = date_map[date_col]
        time_rows = []
        for label, measure in measures.items():
            temp = pd.DataFrame({
                "Date": pd.to_datetime(date_values, errors="coerce"),
                "Value": _safe_numeric(df, measure),
            }).dropna()
            if temp.empty:
                continue
            monthly = temp.set_index("Date")["Value"].resample("ME").sum().rename(label)
            time_rows.append(monthly)
            comparison, _ = _period_comparison(df, date_values, measure)
            if comparison:
                report["tables"][f"MoM - {measure}"] = pd.DataFrame([comparison])
                change = comparison["Change"]
                pct = comparison["Change %"]
                direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
                report["facts"].append(
                    f"{measure} {direction} by {_fmt(abs(change))} "
                    f"({(_fmt(pct) + '%') if pd.notna(pct) else 'percentage change unavailable'}) "
                    f"from {comparison['Previous month']} to {comparison['Latest complete month']}."
                )
                for dimension_type, cols in dimensions.items():
                    # Attribute month-over-month changes to the detected dimensions.
                    for dim in cols[:2]:
                        _, drivers = _period_comparison(df, date_values, measure, dim)
                        if drivers is not None:
                            report["tables"][f"Drivers - {measure} by {dim}"] = drivers.head(30)
        if time_rows:
            time_table = pd.concat(time_rows, axis=1).fillna(0).reset_index()
            time_table.rename(columns={"index": "Month"}, inplace=True)
            report["tables"]["Monthly trends"] = time_table
            report["charts"]["Monthly trends"] = time_table

        if not time_rows:
            report["limitations"].append("A date column was detected, but no usable date/measure pairs were available.")
    else:
        report["limitations"].append("No date column was detected; time-based comparisons are unavailable.")

    if not numeric:
        report["limitations"].append("No numeric measures were detected; quantitative performance analysis is limited.")
    if not dimensions:
        report["limitations"].append("No likely region, product, or customer dimensions were detected.")

    report["facts"].insert(
        0,
        f"Dataset contains {len(df):,} rows and {len(df.columns):,} columns, "
        f"with {report['missing_cells']:,} missing cells and {report['duplicate_rows']:,} duplicate rows."
    )
    return report


def report_context(report):
    """Create a compact, auditable context for Nemotron."""
    parts = [
        "You are explaining a report calculated by Pandas. Treat all numbers below as authoritative.",
        "Do not calculate new figures, infer missing data, claim causation, or introduce facts not listed.",
        "Separate measured facts from hypotheses. Mention limitations where relevant.",
        "Dataset profile: " + str({
            k: report[k] for k in ("rows", "columns", "missing_cells", "duplicate_rows",
                                   "measures", "dimensions", "date_columns", "limitations")
        }),
        "Verified findings:\n" + "\n".join(f"- {x}" for x in report["facts"][:60]),
    ]
    for name, table in list(report["tables"].items())[:15]:
        parts.append(f"Table: {name}\n{table.head(12).to_csv(index=False)}")
    return "\n\n".join(parts)[:24000]


def generate_report_excel(report, ai_narrative=""):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([{
            "Rows": report["rows"],
            "Columns": report["columns"],
            "Missing cells": report["missing_cells"],
            "Duplicate rows": report["duplicate_rows"],
        }]).to_excel(writer, sheet_name="Executive Summary", index=False)
        pd.DataFrame({"Verified finding": report["facts"]}).to_excel(
            writer, sheet_name="Key Findings", index=False
        )
        if ai_narrative:
            pd.DataFrame({"Nemotron narrative": ai_narrative.splitlines()}).to_excel(
                writer, sheet_name="AI Interpretation", index=False
            )
        for index, (name, table) in enumerate(report["tables"].items(), start=1):
            sheet = re.sub(r"[\[\]:*?/\\]", " ", name)[:25].strip() or f"Analysis {index}"
            # Excel sheet names must be unique.
            base, suffix = sheet, 1
            while sheet in writer.book.sheetnames:
                sheet = f"{base[:27]}{suffix}"
                suffix += 1
            table.to_excel(writer, sheet_name=sheet[:31], index=False)
    return output.getvalue()


def generate_report_pdf(report, ai_narrative=""):
    """Create a simple PDF report; requires reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Business Performance Report", styles["Title"]),
        Paragraph(f"Dataset: {report['rows']:,} rows × {report['columns']:,} columns", styles["Normal"]),
        Spacer(1, 8),
        Paragraph("Executive Summary & Data Quality", styles["Heading2"]),
        Paragraph(f"Missing cells: {report['missing_cells']:,} | Duplicate rows: {report['duplicate_rows']:,}", styles["BodyText"]),
    ]
    for fact in report["facts"]:
        story.append(Paragraph("• " + escape(str(fact)), styles["BodyText"]))
        story.append(Spacer(1, 3))
    if ai_narrative:
        story += [Spacer(1, 8), Paragraph("Business Interpretation", styles["Heading2"]),
                  Paragraph(escape(ai_narrative).replace("\n", "<br/>"), styles["BodyText"])]
    for name, frame in report["tables"].items():
        story += [Spacer(1, 8), Paragraph(escape(name), styles["Heading2"])]
        view = frame.head(18).copy().fillna("")
        data = [list(map(str, view.columns))] + view.astype(str).values.tolist()
        data = [row[:7] for row in data]
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26364a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(table)
    doc.build(story)
    return output.getvalue()


def render_business_report(df):
    st.header("📑 Automated Business Performance Report")
    st.caption("Pandas calculates the metrics and drivers. Nemotron explains the verified results.")

    # Build once automatically for each newly uploaded dataset.
    current_signature = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    report_is_new = (
        st.session_state.get("business_report_signature") != current_signature
        or st.session_state.get("business_report") is None
    )
    if report_is_new:
        with st.spinner("Automatically analysing business performance and data quality..."):
            st.session_state["business_report"] = build_business_report(df)
            st.session_state["business_report_signature"] = current_signature
            st.session_state.pop("business_report_ai", None)

    if st.button("Refresh business report", key="generate_business_report"):
        with st.spinner("Recalculating business performance, trends, and contributors..."):
            st.session_state["business_report"] = build_business_report(df)
            st.session_state["business_report_signature"] = current_signature
            st.session_state.pop("business_report_ai", None)

    report = st.session_state.get("business_report")
    if report is None:
        st.info("Generate the report to calculate performance KPIs, regional/product/customer breakdowns, time trends, and contribution drivers.")
        return

    st.subheader("1. Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{report['rows']:,}")
    m2.metric("Columns", f"{report['columns']:,}")
    m3.metric("Missing cells", f"{report['missing_cells']:,}")
    m4.metric("Duplicate rows", f"{report['duplicate_rows']:,}")
    if report["facts"]:
        st.markdown("\n".join(f"- {fact}" for fact in report["facts"][:12]))

    st.subheader("2–6. Performance Analysis")
    if report["tables"].get("Overall performance") is not None:
        st.markdown("**Overall business performance**")
        st.dataframe(report["tables"]["Overall performance"], use_container_width=True)
    for title, frame in report["tables"].items():
        if title == "Overall performance":
            continue
        with st.expander(title):
            st.dataframe(frame, use_container_width=True)
            if title == "Monthly trends":
                chart_frame = frame.set_index("Date") if "Date" in frame.columns else frame
                date_field = chart_frame.index.name or chart_frame.columns[0]
                value_cols = [c for c in chart_frame.columns if c != date_field]
                if value_cols:
                    st.line_chart(chart_frame[value_cols])

    if report["limitations"]:
        st.subheader("Analysis limitations")
        for limitation in report["limitations"]:
            st.warning(limitation)

    st.subheader("7–11. Insights and Recommendations")
    if st.button("Ask Nemotron to explain the calculated report", key="explain_business_report"):
        with st.spinner("Nemotron is interpreting the verified findings..."):
            st.session_state["business_report_ai"] = ask_nemotron(
                "Write an executive business performance report with insights, risks, opportunities, and actionable recommendations. Use only the supplied calculated facts. Clearly label hypotheses and do not claim causation.",
                report_context(report),
            )
    narrative = st.session_state.get("business_report_ai", "")
    if narrative:
        st.markdown(narrative)
    else:
        st.info("Nemotron interpretation is optional. Generate it when you want narrative insights and recommendations.")

    st.subheader("12. Download Report")
    st.download_button(
        "⬇️ Download complete Excel report",
        data=generate_report_excel(report, narrative),
        file_name="business_performance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_business_report_excel",
    )
    try:
        pdf_bytes = generate_report_pdf(report, narrative)
        st.download_button(
            "⬇️ Download PDF report",
            data=pdf_bytes,
            file_name="business_performance_report.pdf",
            mime="application/pdf",
            key="download_business_report_pdf",
        )
    except ImportError:
        st.caption("PDF export requires reportlab. Add `reportlab` to requirements.txt to enable it.")


# =========================================================
# MAIN APPLICATION
# =========================================================

uploaded = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

if uploaded is not None:

    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(
            f"Please upload a file smaller than {MAX_FILE_MB} MB."
        )
        st.stop()

    try:
        df = load_dataset(
            uploaded.getvalue(),
            uploaded.name,
        )

        if df.empty or len(df.columns) == 0:
            st.error("This file has no usable rows or columns.")
            st.stop()

        # Remove leading/trailing spaces from column names.
        df.columns = [
            col.strip() if isinstance(col, str) else col
            for col in df.columns
        ]

        st.success(
            f"Loaded **{uploaded.name}** — "
            f"{len(df):,} rows × {len(df.columns):,} columns"
        )

        # Temporary diagnostics.
        with st.expander("Dataset diagnostics"):
            st.write("**Dataset columns:**", df.columns.tolist())
            st.write(
                "**Dataset data types:**",
                df.dtypes.astype(str).to_dict(),
            )

        # Automated report is a separate workflow; the existing chat analyst remains intact.
        render_business_report(df)


        a, b, c, d = st.columns(4)

        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{len(df.columns):,}")
        c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

        st.download_button(
            "⬇️ Download uploaded dataset as Excel",
            data=dataframe_to_excel_bytes(df, "Dataset"),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_dataset.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_full_dataset",
        )

        # -------------------------------------------------
        # DATA PREVIEW
        # -------------------------------------------------

        with st.expander("Preview data", expanded=True):
            st.dataframe(
                df.head(20),
                use_container_width=True,
            )

        # -------------------------------------------------
        # COLUMN PROFILE
        # -------------------------------------------------

        with st.expander("Column profile"):
            numeric, categorical, dates, _ = get_schema(df)

            profile = pd.DataFrame([
                {
                    "Column": str(col),
                    "Type": str(df[col].dtype),
                    "Detected type": (
                        "Numeric" if col in numeric
                        else "Date" if col in dates
                        else "Category"
                    ),
                    "Missing": int(df[col].isna().sum()),
                    "Unique": int(df[col].nunique(dropna=True)),
                    "Examples": ", ".join(
                        df[col].dropna().astype(str).head(3).tolist()
                    ),
                }
                for col in df.columns
            ])

            st.dataframe(
                profile,
                use_container_width=True,
            )

            st.write(
                "**Numeric columns:**",
                ", ".join(map(str, numeric)) or "None",
            )

            st.write(
                "**Category columns:**",
                ", ".join(map(str, categorical)) or "None",
            )

            st.write(
                "**Date columns:**",
                ", ".join(map(str, dates)) or "None",
            )

        # -------------------------------------------------
        # NUMERIC EXPLORER
        # -------------------------------------------------

        with st.expander("Explore a numeric column"):
            numeric, _, _, _ = get_schema(df)

            if numeric:
                selected = st.selectbox(
                    "Numeric column",
                    numeric,
                )

                values = clean_numeric_series(df[selected])

                st.dataframe(
                    values.describe().to_frame(),
                    use_container_width=True,
                )

                chart = st.radio(
                    "Chart",
                    ["Histogram", "Box plot"],
                    horizontal=True,
                )

                if chart == "Histogram":
                    fig = px.histogram(
                        x=values,
                        title=f"Distribution of {selected}",
                        labels={"x": selected},
                    )
                else:
                    fig = px.box(
                        y=values,
                        title=f"Box plot of {selected}",
                        labels={"y": selected},
                    )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="numeric_explore_chart",
                )

            else:
                st.info("No numeric columns detected.")

        # -------------------------------------------------
        # CHAT INTERFACE
        # -------------------------------------------------

        st.header("💬 Ask the Data Analyst")

        st.caption(
            "Examples: “top 5 products by Sales”, "
            "“compare Sales by Ship Mode”, "
            "“show monthly Sales trend”, "
            "“summarize dataset”."
        )

        file_signature = hashlib.sha256(
            uploaded.getvalue()
        ).hexdigest()

        if st.session_state.get("dataset_signature") != file_signature:
            st.session_state.dataset_signature = file_signature
            st.session_state.chat_history = []

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display previous messages.
        for history_index, item in enumerate(
            st.session_state.chat_history
        ):
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

                if item.get("table") is not None:
                    st.dataframe(
                        item["table"],
                        use_container_width=True,
                    )

                    st.download_button(
                        "⬇️ Download these results as Excel",
                        data=dataframe_to_excel_bytes(
                            item["table"],
                            "Analysis",
                        ),
                        file_name=f"analysis_results_{history_index}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_history_{history_index}",
                    )

                if item.get("figure") is not None:
                    st.plotly_chart(
                        item["figure"],
                        use_container_width=True,
                        key=f"history_plot_{history_index}",
                    )

        question = st.chat_input(
            "Ask a question about the uploaded dataset…"
        )

        if question:

            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                with st.spinner("Analyzing your data with AI…"):
                    output = answer_question(df, question)

                    # Give Nemotron the real calculated results.
                    try:
                        context_parts = [
                            "Original analysis result:",
                            str(output["answer"]),
                        ]

                        table = output.get("table")

                        if table is not None:
                            context_parts.append(
                                "Calculated result table:\n"
                                + table.head(30).to_csv(index=False)
                            )

                        ai_explanation = ask_nemotron(
                            question,
                            "\n\n".join(context_parts),
                        )

                        output["answer"] = (
                            output["answer"]
                            + "\n\n---\n\n"
                            + "### 🤖 Nemotron's explanation\n"
                            + ai_explanation
                        )

                    except Exception as exc:
                        st.warning(
                            "The analysis completed, but the AI "
                            f"explanation failed: {exc}"
                        )

                st.markdown(output["answer"])

                # UPDATED: Display the current table only once.
                # The duplicate current-result download buttons
                # have been removed.

                if output.get("table") is not None:
                    st.dataframe(
                        output["table"],
                        use_container_width=True,
                    )

                if output.get("figure") is not None:
                    st.plotly_chart(
                        output["figure"],
                        use_container_width=True,
                        key="current_answer_plot",
                    )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": output["answer"],
                "table": output.get("table"),
                "figure": output.get("figure"),
            })

        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    except Exception as exc:
        st.error(
            f"Could not read or analyze this file: {exc}"
        )

else:
    st.markdown("### What you can ask")

    st.write("- What is the total Sales?")
    st.write("- Which are the top 5 products by Sales?")
    st.write("- Compare Sales by Ship Mode.")
    st.write("- Show monthly Sales trends.")
    st.write("- Which columns have missing values?")
    st.write("- Show correlations between numeric columns.")
    st.write("- Summarize this dataset.")

