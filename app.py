import re
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Analyst")
st.caption("Upload a CSV or Excel file, then ask questions in everyday language.")
st.info(
    "Free deployment mode: calculations run in Python without a paid AI API or a local Ollama server. "
    "This version uses a reusable question interpreter for common data-analysis tasks."
)

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
    """Parse dates conservatively; return None if there are too few valid dates."""
    parsed = pd.to_datetime(series, errors="coerce")
    if len(series.dropna()) == 0:
        return None
    if parsed.notna().mean() >= 0.75:
        return parsed
    return None

def get_schema(df):
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    date_map = {}
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_map[c] = pd.to_datetime(df[c], errors="coerce")
        elif any(token in str(c).lower() for token in ("date", "time", "timestamp")):
            parsed = parse_dates(df[c])
            if parsed is not None:
                date_map[c] = parsed
    dates = list(date_map)
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    return numeric, categorical, dates, date_map

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

def choose_category(question, categorical, explicit):
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
    for triggers, tokens in hints:
        if any(t in q for t in triggers):
            for c in categorical:
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
    category = choose_category(question, categorical, explicit)
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
        freq = "D" if "daily" in q or "by day" in q else "W" if "weekly" in q or "by week" in q else "Y" if "yearly" in q or "by year" in q else "Q" if "quarter" in q else "M"
        temp = temp.set_index("date")["value"].resample(freq).sum().rename(measure).reset_index()
        fig = px.line(temp, x="date", y=measure, markers=True, title=f"{measure} over time")
        return result(f"### Trend: {measure} over time\nAggregated using the sum at the requested time frequency.", temp, fig)
   freq = "D" if "daily" in q or "by day" in q else "W" if "weekly" in q or "by week" in q else "YE" if "yearly" in q or "by year" in q else "QE" if "quarter" in q else "ME"
    rank_match = re.search(r"\b(top|bottom)\s+(\d+)\b", q)
    is_rank = rank_match or any(x in q for x in ("highest", "lowest", "rank", "best", "worst", "most sales", "least sales"))
    if is_rank:
        if measure is None or category is None:
            return result(
                "To rank items, I need a numeric measure and a category column. "
                f"Numeric columns: {', '.join(map(str, numeric)) or 'none'}. "
                f"Category columns: {', '.join(map(str, categorical)) or 'none'}. "
                "Example: 'Which are the top 5 products by Sales?'"
            )
        agg = df.groupby(category, dropna=False)[measure].sum(min_count=1).reset_index()
        descending = not ("bottom" in q or "lowest" in q or "worst" in q or "least" in q)
        agg = agg.sort_values(measure, ascending=not descending, na_position="last")
        limit = min(int(rank_match.group(2)), 100) if rank_match else 10
        table = agg.head(limit)
        label = "Top" if descending else "Bottom"
        fig = px.bar(table, x=category, y=measure, title=f"{label} {len(table)} {category} by {measure}")
        return result(f"### {label} {len(table)} {category} by {measure}\nValues are summed by category.", table, fig)

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
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns detected.")

        st.header("💬 Ask the Data Analyst")
        st.caption("Examples: “top 5 products by Sales”, “compare Sales by Ship Mode”, “show monthly Sales trend”, “summarize dataset”.")
        file_signature = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("dataset_signature") != file_signature:
            st.session_state.dataset_signature = file_signature
            st.session_state.chat_history = []
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])
                if item.get("table") is not None:
                    st.dataframe(item["table"], use_container_width=True)
                if item.get("figure") is not None:
                    st.plotly_chart(item["figure"], use_container_width=True)

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
                if output.get("figure") is not None:
                    st.plotly_chart(output["figure"], use_container_width=True)
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
