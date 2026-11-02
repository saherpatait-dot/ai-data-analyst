import re
import hashlib
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst")
st.caption("Upload a CSV or Excel file.")
st.info("Please select the column names in your dataset for accurate analysis.")
st.write("Dataset columns:", df.columns.tolist())
st.write("Dataset data types:", df.dtypes.astype(str).to_dict())

MAX_FILE_MB = 200

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO
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


def parse_dates(series):
    """Detect dates stored as text in CSV or Excel files."""
    if len(series.dropna()) == 0:
        return None

    try:
        parsed = pd.to_datetime(
            series, errors="coerce", format="mixed"
        )
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce")

    valid_ratio = parsed.notna().sum() / max(series.notna().sum(), 1)

    if valid_ratio >= 0.75:
        return parsed

    return None


def get_schema(df):
    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]

    date_map = {}

    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_map[c] = pd.to_datetime(df[c], errors="coerce")
        elif any(
            token in str(c).lower()
            for token in ("date", "time", "timestamp")
        ):
            parsed = parse_dates(df[c])
            if parsed is not None:
                date_map[c] = parsed

    dates = list(date_map.keys())
    categorical = [
        c for c in df.columns
        if c not in numeric and c not in dates
    ]

    return numeric, categorical, dates, date_map


def dataframe_to_excel_bytes(dataframe: pd.DataFrame, sheet_name: str = "Analysis") -> bytes:
    """Create an in-memory Excel workbook for a dataframe."""
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Analysis")
    return output.getvalue()


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

def mentioned_columns(question, columns):
    q = norm(question)
    hits = []
    for col in columns:
        n = norm(col)
        if n and re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", q):
            hits.append(col)
    return sorted(hits, key=lambda c: len(norm(c)), reverse=True)

def choose_measure(question, numeric, explicit):
    for col in explicit:
        if col in numeric:
            return col
    q = norm(question)
    # Semantic names are useful when the question says "sales" but the column is "Sales Amount".
    hints = [
        (("sales", "revenue", "turnover", "income"), ("sales", "revenue", "turnover", "amount")),
        (("profit", "earnings", "margin"), ("profit", "earnings", "margin")),
        (("quantity", "units", "items sold", "volume"), ("quantity", "units", "volume")),
        (("discount",), ("discount",)),
        (("cost", "expense", "spend"), ("cost", "expense", "spend")),
        (("price",), ("price",)),
        (("age",), ("age",)),
        (("score", "rating"), ("score", "rating")),
    ]
    for triggers, tokens in hints:
        if any(t in q for t in triggers):
            for c in numeric:
                if any(t in norm(c).split() for t in tokens) or any(t in norm(c) for t in tokens):
                    return c
    if len(numeric) == 1:
        return numeric[0]
    return None

def choose_category(question, categorical, explicit, numeric=None):
    for col in explicit:
        if col in categorical:
            return col
    q = norm(question)
    hints = [
        (("product", "products", "item", "items", "sku"), ("product", "item", "sku")),
        (("customer", "customers", "client", "clients"), ("customer", "client", "buyer")),
        (("region", "regional", "area", "territory"), ("region", "area", "territory", "state", "country")),
        (("ship mode", "shipping", "shipment"), ("ship mode", "ship", "shipping", "shipment")),
        (("category", "categories"), ("category",)),
        (("segment", "segments"), ("segment",)),
        (("department", "departments"), ("department", "dept")),
        (("city", "cities"), ("city",)),
        (("state", "states"), ("state",)),
        (("country", "countries"), ("country",)),
    ]
    # Numeric identifier columns (e.g. DEPARTMENT_ID) can be valid grouping categories.
    numeric = numeric or []
    groupable = list(categorical) + [
        c for c in numeric
        if any(token in norm(c) for token in ("id", "code", "number"))
    ]
    for triggers, tokens in hints:
        if any(t in q for t in triggers):
            for c in groupable:
                if any(t in norm(c) for t in tokens):
                    return c
    if len(categorical) == 1:
        return categorical[0]
    return None

def answer_question(df, question):
    q = question.lower().strip()
    numeric, categorical, dates, date_map = get_schema(df)
    explicit = mentioned_columns(question, list(df.columns))
    measure = choose_measure(question, numeric, explicit)
    category = choose_category(question, categorical, explicit, numeric)
    # Numeric columns can also be grouping categories (e.g. Price range = 1, 2, 3, 4).
    if category is None:
        category = next((c for c in explicit if c != measure and c not in dates), None)
    date_col = next((c for c in explicit if c in dates), None)
    if date_col is None and len(dates) == 1:
        date_col = dates[0]

    def result(answer, table=None, fig=None):
        return {"answer": answer, "table": table, "figure": fig}

    if any(x in q for x in ("help", "what can you do", "example questions")):
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

    if any(x in q for x in ("overview", "summarize dataset", "dataset summary", "summary of dataset", "data quality", "missing values", "duplicates", "how many rows", "how many columns")):
        missing = df.isna().sum()
        table = pd.DataFrame({
            "Column": [str(c) for c in df.columns],
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Missing values": [int(missing[c]) for c in df.columns],
            "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        return result(
            f"### Dataset overview\n- **Rows:** {len(df):,}\n- **Columns:** {len(df.columns):,}\n"
            f"- **Missing cells:** {int(missing.sum()):,}\n- **Duplicate rows:** {int(df.duplicated().sum()):,}",
            table
        )

    if any(x in q for x in ("correlation", "correlations", "correlate", "relationship between")):
        if len(numeric) < 2:
            return result("Correlation analysis needs at least two numeric columns.")
        corr = df[numeric].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix")
        return result("### Correlation matrix\nValues range from -1 to +1. Correlation does not prove causation.", corr, fig)

    if any(x in q for x in ("trend", "over time", "monthly", "daily", "weekly", "yearly", "by month", "by year", "quarterly")):
        if measure is None or date_col is None:
            return result(
                "To make a trend chart, I need a numeric measure and a date column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Date columns detected: {', '.join(map(str, dates)) or 'none'}. "
                "Try naming both, e.g. 'Show monthly Sales by Order Date'."
            )
        temp = pd.DataFrame({"date": date_map[date_col], "value": pd.to_numeric(df[measure], errors="coerce")}).dropna()
        if temp.empty:
            return result(f"No valid dates and numeric values were found for {date_col} and {measure}.")
        freq = "D" if "daily" in q or "by day" in q else "W" if "weekly" in q or "by week" in q else "YE" if "yearly" in q or "by year" in q else "QE" if "quarter" in q else "ME"
        temp = temp.set_index("date")["value"].resample(freq).sum().rename(measure).reset_index()
        fig = px.line(temp, x="date", y=measure, markers=True, title=f"{measure} over time")
        return result(f"### Trend: {measure} over time\nAggregated using the sum at the requested time frequency.", temp, fig)
    rank_match = re.search(r"\b(top|bottom)\s+(\d+)\b", q)
    is_rank = bool(rank_match) or any(x in q for x in ("highest", "lowest", "rank", "best", "worst", "most sales", "least sales"))
    count_intent = any(x in q for x in ("number of", "count of", "count the", "how many", "most restaurants", "fewest restaurants", "number of restaurants"))

    # Count records by a requested group, including phrasing such as "how many employees in each department".
    if count_intent and category is not None and any(x in q for x in ("each", "per", "by", "in every")):
        count_col = category
        table = df.groupby(count_col, dropna=False).size().rename("Count").reset_index()
        table = table.sort_values("Count", ascending=False, na_position="last")
        fig = px.bar(table.head(40), x=count_col, y="Count", title=f"Employee/record count by {count_col}")
        return result(f"### Record count by {count_col}", table, fig)

    # Direct row-level ranking: "who has the highest salary?" / "top 5 employees by salary".
    row_level_intent = any(x in q for x in ("who has", "which employee", "employees by", "employee by", "people by", "person by"))
    if measure is not None and (row_level_intent or (is_rank and category is None)):
        descending = not any(x in q for x in ("bottom", "lowest", "worst", "least"))
        table = df.sort_values(measure, ascending=not descending, na_position="last")
        limit = min(int(rank_match.group(2)), 100) if rank_match else 1 if any(x in q for x in ("who has", "which employee")) else 10
        table = table.head(limit)
        label = "Highest" if descending and limit == 1 else "Lowest" if not descending and limit == 1 else "Top" if descending else "Bottom"
        fig = px.bar(table, x=measure, y=table.index.astype(str), orientation="h", title=f"{label} {limit} records by {measure}")
        return result(f"### {label} {len(table)} records by {measure}", table, fig)

    if is_rank and count_intent and category is not None:
        count_col = category
        agg = df.groupby(count_col, dropna=False).size().rename("Count").reset_index()
        descending = not ("bottom" in q or "lowest" in q or "worst" in q or "fewest" in q or "least" in q)
        agg = agg.sort_values("Count", ascending=not descending, na_position="last")
        limit = min(int(rank_match.group(2)), 100) if rank_match else 10
        table = agg.head(limit)
        label = "Top" if descending else "Bottom"
        fig = px.bar(table, x=count_col, y="Count", title=f"{label} {len(table)} {count_col} by number of records")
        return result(f"### {label} {len(table)} {count_col} by number of records", table, fig)
    if is_rank:
        if measure is None or category is None:
            return result(
                "To rank items, I need a numeric measure and a category column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Category columns: {', '.join(map(str, categorical)) or 'none'}. "
                "Example: 'Which are the top 5 products by Sales?'"
            )
        # Respect the aggregation requested in ranking questions.
        # "Highest average salary by department" must rank means, not sums.
        agg_name = "mean" if any(x in q for x in ("average", "mean", "avg")) else "sum"
        agg = df.groupby(category, dropna=False)[measure].agg(agg_name).reset_index()
        descending = not ("bottom" in q or "lowest" in q or "worst" in q or "least" in q)
        agg = agg.sort_values(measure, ascending=not descending, na_position="last")
        limit = min(int(rank_match.group(2)), 100) if rank_match else 1 if any(x in q for x in ("highest", "lowest", "best", "worst")) else 10
        table = agg.head(limit)
        label = "Highest" if descending and limit == 1 else "Lowest" if not descending and limit == 1 else "Top" if descending else "Bottom"
        fig = px.bar(table, x=category, y=measure, title=f"{label} {len(table)} {category} by {agg_name} {measure}")
        return result(f"### {label} {len(table)} {category} by {agg_name} {measure}\\nValues use {agg_name} aggregation by category.", table, fig)

    if any(x in q for x in ("compare", "comparison", "breakdown", "by category", "by region", "by product", "across")):
        if measure is None or category is None:
            return result(
                "To compare groups, I need a numeric measure and a category column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Category columns: {', '.join(map(str, categorical)) or 'none'}."
            )
        agg_name = "mean" if any(x in q for x in ("average", "mean")) else "sum"
        table = df.groupby(category, dropna=False)[measure].agg(agg_name).reset_index()
        table = table.sort_values(measure, ascending=False, na_position="last")
        fig = px.bar(table.head(40), x=category, y=measure, title=f"{agg_name.title()} {measure} by {category}")
        return result(f"### {agg_name.title()} {measure} by {category}", table, fig)

    if any(x in q for x in ("missing", "null", "empty cells")):
        table = df.isna().sum().rename("Missing values").to_frame()
        table = table[table["Missing values"] > 0].sort_values("Missing values", ascending=False)
        return result("### Missing values\nNo missing values found." if table.empty else "### Missing values by column", table)

    if any(x in q for x in ("duplicate", "duplicates")):
        return result(f"### Duplicate rows\nFound **{int(df.duplicated().sum()):,}** duplicate row(s). No rows have been removed.")

    if measure is not None and any(x in q for x in ("total", "sum", "average", "mean", "median", "minimum", "maximum", "max", "min", "standard deviation", "describe", "summary", "how much", "how many")):
        s = pd.to_numeric(df[measure], errors="coerce").dropna()
        if s.empty:
            return result(f"No valid numeric values found in {measure}.")
        stats = pd.DataFrame({"Statistic": ["Count", "Sum", "Mean", "Median", "Minimum", "Maximum", "Standard deviation"], "Value": [s.count(), s.sum(), s.mean(), s.median(), s.min(), s.max(), s.std()]})
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

    if any(x in q for x in ("distribution", "histogram", "spread")):
        if measure is None:
            return result("Please name the numeric column whose distribution you want to inspect.")
        fig = px.histogram(df, x=measure, title=f"Distribution of {measure}")
        return result(f"### Distribution of {measure}", df[measure].describe().to_frame(), fig)

    if any(x in q for x in ("total", "sum", "average", "mean", "median")) and not explicit:
        return result(
            "I couldn't find a column in this dataset that matches the measure in your question. "
            f"Available numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
            "Try asking about one of these columns."
        )

    return result(
        "I couldn't confidently identify the analysis you want. Try **help** for examples, "
        "or name the measure and category explicitly (for example, 'top 5 products by Sales').\n\n"
        "This free-hosting version uses a safe question interpreter rather than a large language model, "
        "so it supports common analysis requests but not every possible conversational question."
    )

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        st.error(f"Please upload a file smaller than {MAX_FILE_MB} MB.")
        st.stop()
    try:
        df = load_dataset(uploaded.getvalue(), uploaded.name)
        if df.empty or len(df.columns) == 0:
            st.error("This file has no usable rows or columns.")
            st.stop()
        st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns):,} columns")
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

        with st.expander("Preview data", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)
        with st.expander("Column profile"):
            numeric, categorical, dates, _ = get_schema(df)
            profile = pd.DataFrame([{
                "Column": str(col),
                "Type": str(df[col].dtype),
                "Missing": int(df[col].isna().sum()),
                "Unique": int(df[col].nunique(dropna=True)),
                "Examples": ", ".join(df[col].dropna().astype(str).head(3).tolist())
            } for col in df.columns])
            st.dataframe(profile, use_container_width=True)
            st.write("**Numeric columns:**", ", ".join(map(str, numeric)) or "None")
            st.write("**Category columns:**", ", ".join(map(str, categorical)) or "None")
            st.write("**Date columns:**", ", ".join(map(str, dates)) or "None")

        with st.expander("Explore a numeric column"):
            numeric, _, _, _ = get_schema(df)
            if numeric:
                selected = st.selectbox("Numeric column", numeric)
                st.dataframe(df[selected].describe().to_frame(), use_container_width=True)
                chart = st.radio("Chart", ["Histogram", "Box plot"], horizontal=True)
                fig = px.histogram(df, x=selected, title=f"Distribution of {selected}") if chart == "Histogram" else px.box(df, y=selected, title=f"Box plot of {selected}")
                st.plotly_chart(fig, use_container_width=True, key="numeric_explore_chart")
            else:
                st.info("No numeric columns detected.")

        st.header("💬 Ask the Data Analyst")
        st.caption("Examples: “top 5 products by Sales”, “compare Sales by Ship Mode”, “show monthly Sales trend”, “summarize dataset”.")
        file_signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
        if st.session_state.get("dataset_signature") != file_signature:
            st.session_state.dataset_signature = file_signature
            st.session_state.chat_history = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for history_index, item in enumerate(st.session_state.chat_history):
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("table") is not None:
                    st.dataframe(item["table"], use_container_width=True)
                    st.download_button(
                        "⬇️ Download these results as Excel",
                        data=dataframe_to_excel_bytes(item["table"], "Analysis"),
                        file_name=f"analysis_results_{history_index}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_history_{history_index}",
                    )
                if item.get("figure") is not None:
                    st.plotly_chart(item["figure"], use_container_width=True, key=f"history_plot_{history_index}")

        question = st.chat_input("Ask a question about the uploaded dataset…")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data…"):
                    output = answer_question(df, question)
                st.markdown(output["answer"])
                if output.get("table") is not None:
                    st.dataframe(output["table"], use_container_width=True)
                    st.download_button(
                        "⬇️ Download these results as Excel",
                        data=dataframe_to_excel_bytes(output["table"], "Analysis"),
                        file_name="analysis_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_current_{len(st.session_state.chat_history)}",
                    )
                if output.get("figure") is not None:
                    st.plotly_chart(output["figure"], use_container_width=True, key="current_answer_plot")
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
        st.error(f"Could not read or analyze this file: {exc}")
else:
    st.markdown("### What you can ask")
    st.write("- What is the total Sales?")
    st.write("- Which are the top 5 products by Sales?")
    st.write("- Compare Sales by Ship Mode.")
    st.write("- Show monthly Sales trends.")
    st.write("- Which columns have missing values?")
    st.write("- Show correlations between numeric columns.")
    st.write("- Summarize this dataset.")
