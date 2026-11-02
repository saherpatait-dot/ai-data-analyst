```python
import re
import hashlib
from io import BytesIO

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st


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

    # Treat common missing-value strings as missing.
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
            # Numeric identifiers and codes should usually remain
            # categories rather than analytical measures.
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


def choose_measure(question, numeric, explicit):
    """Select the numeric measure requested by the user."""

    # Prefer an explicitly named numeric column.
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

        # Temporary diagnostics to help verify column detection.
        with st.expander("Dataset diagnostics"):
            st.write("**Dataset columns:**", df.columns.tolist())
            st.write(
                "**Dataset data types:**",
                df.dtypes.astype(str).to_dict(),
            )

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

                with st.spinner("Analyzing your data…"):
                    output = answer_question(df, question)

                st.markdown(output["answer"])

                if output.get("table") is not None:
                    st.dataframe(
                        output["table"],
                        use_container_width=True,
                    )

                    st.download_button(
                        "⬇️ Download these results as Excel",
                        data=dataframe_to_excel_bytes(
                            output["table"],
                            "Analysis",
                        ),
                        file_name="analysis_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_current_{len(st.session_state.chat_history)}",
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
```
