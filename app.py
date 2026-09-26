"""ChoiceMap Streamlit application.

Run: streamlit run app.py
Requires engine.py and visual.py alongside this file.
Add to requirements.txt: streamlit>=1.45,<2
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import plotly.io as pio
import polars as pl
import streamlit as st

from engine import Config, run
from visual import build_figures, load_run

BASE = Path(__file__).resolve().parent
SAMPLE = BASE / "sample_data" / "instacart_yogurt.parquet"
MAX_UPLOAD_MB = 300

st.set_page_config(
    page_title="ChoiceMap | Customer choice intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """<style>
.block-container {padding-top: 2.2rem; max-width: 1450px;}
h1, h2, h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: #f5f8fc; border: 1px solid #e1e8f2;
 border-radius: 10px; padding: 14px 17px;}
div[data-testid="stAlert"] {border-radius: 8px;}
</style>""",
    unsafe_allow_html=True,
)


def session_directory() -> Path:
    if "workspace" not in st.session_state:
        st.session_state.workspace = tempfile.mkdtemp(prefix="choicemap_")
    return Path(st.session_state.workspace)


def input_columns(source: str, uploaded) -> list[str]:
    if source == "Sample dataset":
        if not SAMPLE.is_file():
            return []
        return list(pl.read_parquet_schema(SAMPLE))
    if uploaded is None:
        return []
    uploaded.seek(0)
    try:
        if uploaded.name.lower().endswith(".parquet"):
            return list(pl.read_parquet_schema(uploaded))
        return pl.read_csv(uploaded, n_rows=1, infer_schema_length=100).columns
    finally:
        uploaded.seek(0)


def save_upload(uploaded, destination: Path) -> str:
    digest = hashlib.sha256()
    uploaded.seek(0)
    with destination.open("wb") as handle:
        while True:
            block = uploaded.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            handle.write(block)
    uploaded.seek(0)
    return digest.hexdigest()


def column_picker(title: str, columns: list[str], default: str, *, optional=False):
    options = (["(none)"] if optional else []) + columns
    proposed = default if default in columns else ("(none)" if optional else options[0])
    selected = st.selectbox(title, options, index=options.index(proposed))
    return None if selected == "(none)" else selected


st.title("ChoiceMap")
st.caption("Customer choice intelligence from transaction histories")
st.write(
    "Build a SKU hierarchy from customer overlap. Inspect basket relationships and "
    "the stability of each tree branch separately."
)

with st.sidebar:
    st.header("Data and model")
    source = st.radio("Data source", ["Sample dataset", "Upload data"], horizontal=False)
    uploaded = None
    if source == "Upload data":
        uploaded = st.file_uploader(
            "Transaction file",
            type=["parquet", "csv"],
            help=f"CSV or Parquet, up to {MAX_UPLOAD_MB} MB.",
        )
        if uploaded is not None and uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB app limit.")
            uploaded = None
    elif not SAMPLE.is_file():
        st.warning("Sample missing. Add sample_data/instacart_yogurt.parquet or upload data.")
    try:
        columns = input_columns(source, uploaded)
    except Exception as exc:
        st.error(f"Could not inspect schema: {exc}")
        columns = []

    ready = bool(columns) and (source == "Sample dataset" or uploaded is not None)
    if ready:
        with st.form("build_form", clear_on_submit=False):
            st.subheader("Column mapping")
            user_col = column_picker("Customer ID", columns, "user_id")
            order_col = column_picker("Order ID", columns, "order_id")
            product_col = column_picker("Product ID", columns, "product_id")
            name_col = column_picker("Product name", columns, "product_name", optional=True)
            sequence_col = column_picker(
                "Customer order sequence", columns, "order_number", optional=True
            )
            category_col = column_picker(
                "Category column",
                columns,
                "aisle" if source == "Sample dataset" else "",
                optional=True,
            )
            category_value = None
            if category_col is not None:
                category_value = st.text_input(
                    "Category value",
                    value="yogurt"
                    if source == "Sample dataset" and category_col == "aisle"
                    else "",
                    help="Leave blank to include all rows; choose a single category for interpretable trees.",
                )
                if not category_value.strip():
                    category_value = None
            st.subheader("Model settings")
            min_buyers = st.number_input(
                "Minimum buyers per SKU", min_value=1, max_value=100000, value=30
            )
            max_skus = st.slider(
                "Maximum SKUs",
                min_value=2,
                max_value=250,
                value=80,
                help="Pair tables and bootstrap work grow rapidly with the SKU count.",
            )
            bootstrap = st.slider(
                "Customer bootstrap replicates",
                min_value=0,
                max_value=100,
                value=20,
                help="Start with 5 while iterating; increase for final reporting.",
            )
            method = st.selectbox("Linkage method", ["average", "complete", "single"])
            seed = st.number_input("Random seed", min_value=0, value=42, step=1)
            build = st.form_submit_button(
                "Build choice tree", type="primary", use_container_width=True
            )
    else:
        build = False

if build:
    ids = [user_col, order_col, product_col]
    if len(set(ids)) != len(ids):
        st.error("Customer, order and product IDs must use distinct columns.")
    elif sequence_col in ids or (name_col is not None and name_col in ids):
        st.error("Sequence and product-name columns cannot reuse an ID column.")
    elif category_col is not None and category_value is None:
        st.warning(
            "No category filter is set. Verify that the uploaded file contains only one category."
        )
        # Unfiltered input is allowed, subject to the engine's input-row guard.
    else:
        workspace = session_directory()
        staging = workspace / "staging"
        staging.mkdir(exist_ok=True)
        try:
            if source == "Sample dataset":
                input_path = SAMPLE
                source_id = "sample:" + str(SAMPLE.resolve()) + ":" + str(SAMPLE.stat().st_mtime_ns)
            else:
                ext = ".parquet" if uploaded.name.lower().endswith(".parquet") else ".csv"
                input_path = staging / f"uploaded{ext}"
                source_id = "upload:" + save_upload(uploaded, input_path)
            # Never write untrusted uploaded filenames to disk.
            config = Config(
                input_path=str(input_path),
                out=str(workspace / "model"),
                user_col=user_col,
                order_col=order_col,
                product_col=product_col,
                name_col=name_col or "product_name",
                sequence_col=sequence_col or "order_number",
                category_col=category_col,
                category_value=category_value,
                min_buyers=int(min_buyers),
                max_skus=int(max_skus),
                bootstrap=int(bootstrap),
                linkage_method=method,
                seed=int(seed),
                switching=sequence_col is not None,
            )
            # Engine writes fixed filenames; clear prior results to prevent stale artifacts.
            shutil.rmtree(workspace / "model", ignore_errors=True)
            with st.spinner("Building SKU metrics, choice distances and branch support…"):
                report = run(config)
            st.session_state.result = {
                "directory": str(workspace / "model"),
                "source_id": source_id,
                "report": report,
            }
            st.success("Model built. Use the tabs below to inspect its evidence.")
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"Build failed: {exc}")

result = st.session_state.get("result")
if result is None:
    st.info("Select the sample dataset or upload transactions, then build a model to begin.")
    st.stop()

folder = Path(result["directory"])
if not (folder / "run.json").exists():
    st.warning("Run files are unavailable. Rebuild the model.")
    st.stop()

report = result["report"]
with st.sidebar:
    st.divider()
    st.header("Display settings")
    cut = st.slider(
        "Choice distance cut",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="Recolors tree groups; does not refit the model.",
    )
    search = st.text_input("Find SKU", placeholder="Product ID or name")
    max_labels = st.slider("Visible tree labels", min_value=10, max_value=120, value=55)
    optimal_order = st.checkbox(
        "Optimize leaf ordering", value=False, help="Can be slow; changes only branch orientation."
    )

st.caption(
    f"Last built run · {report['retained_skus']:,} SKUs · "
    f"{report['runtime_seconds']:.1f} s engine runtime"
)
if source == "Upload data" and uploaded is not None:
    st.caption(
        "A previous result remains visible until you click Build choice tree again. "
        "Check that its settings correspond to your current upload."
    )

try:
    data = load_run(folder)
except Exception as exc:
    st.error(f"Could not load run outputs: {exc}")
    st.stop()

sku = data["sku"]
names = data["names"]
ids = data["ids"]
positions = {product_id: index for index, product_id in enumerate(ids)}
options = list(range(len(ids)))


def label(index: int) -> str:
    text = names[index]
    return f"{text[:72]}{'…' if len(text) > 72 else ''} · {ids[index]}"


summary, tree_tab, pairs_tab, exports = st.tabs(
    ["Overview", "Choice tree", "Pair explorer", "Downloads"]
)

with summary:
    cols = st.columns(4)
    cols[0].metric("Category customers", f"{report['category_customers']:,}")
    cols[1].metric("Retained SKUs", f"{report['retained_skus']:,}")
    cols[2].metric("Customer coverage", f"{report['customer_coverage']:.1%}")
    fit = report.get("cophenetic_correlation")
    cols[3].metric("Tree distance fit", "Not defined" if fit is None else f"{fit:.2f}")
    st.markdown("### Reading the model")
    st.write(
        "Customer Jaccard measures how often buyers of either SKU have bought both "
        "across their observed histories. A shorter tree distance indicates more "
        "shared buyers. Order Jaccard and basket lift describe same-order association "
        "and do not determine the choice tree."
    )
    st.dataframe(
        sku.sort("buyers", descending=True).head(25), use_container_width=True, hide_index=True
    )
    st.caption(
        "Coverage refers to customers who bought at least one retained SKU. "
        "The tree does not estimate lost sales or causal substitution."
    )

with tree_tab:
    try:
        tree, heatmap = build_figures(
            folder,
            cut=cut,
            search=search or None,
            max_labels=max_labels,
            optimal_order=optimal_order,
        )
        st.plotly_chart(tree, use_container_width=True, config={"displaylogo": False})
        st.caption(
            "Branch opacity and dotted lines indicate whole-customer bootstrap support. "
            "When bootstrap is zero, stability has not been estimated."
        )
        st.plotly_chart(heatmap, use_container_width=True, config={"displaylogo": False})
    except Exception as exc:
        st.error(f"Could not render visuals: {exc}")

with pairs_tab:
    st.markdown("### Inspect a SKU pair")
    one, two = st.columns(2)
    with one:
        first = st.selectbox("First SKU", options, format_func=label, key="first_sku")
    with two:
        second = st.selectbox(
            "Second SKU",
            options,
            index=min(1, len(options) - 1),
            format_func=label,
            key="second_sku",
        )
    if first == second:
        st.info("Select two different SKUs.")
    else:
        a_id, b_id = ids[first], ids[second]
        pair = (
            pl.scan_parquet(folder / "pair_metrics.parquet")
            .filter((pl.col("product_id_a") == a_id) & (pl.col("product_id_b") == b_id))
            .collect()
        )
        if pair.height:
            row = pair.row(0, named=True)
            metrics = st.columns(4)
            metrics[0].metric("Customer Jaccard", f"{row['customer_jaccard']:.3f}")
            metrics[1].metric("Shared customers", f"{row['shared_customers']:,}")
            metrics[2].metric("Order Jaccard", f"{row['order_jaccard']:.3f}")
            metrics[3].metric("Basket lift", f"{row['basket_lift']:.2f}")
            st.dataframe(pair, hide_index=True, use_container_width=True)
            st.caption(
                "Directional confidence and switching weights use the sorted "
                "product_id_a/product_id_b order shown in the table. An exploratory "
                "q-value is not evidence of causal substitution."
            )
    st.markdown("### Highest shared-customer pairs")
    top = (
        pl.scan_parquet(folder / "pair_metrics.parquet")
        .sort("shared_customers", descending=True)
        .limit(30)
        .collect()
    )
    st.dataframe(
        top.select(
            [
                "product_id_a",
                "product_id_b",
                "shared_customers",
                "customer_jaccard",
                "order_jaccard",
                "basket_lift",
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

with exports:
    st.write(
        "Download the current run's model outputs. HTML includes the display settings "
        "currently selected in the sidebar; model metrics do not change."
    )
    st.download_button(
        "Download run metadata (JSON)",
        data=(folder / "run.json").read_bytes(),
        file_name="choicemap_run.json",
        mime="application/json",
        on_click="ignore",
    )
    for filename in ["sku_metrics.parquet", "pair_metrics.parquet", "tree_nodes.parquet"]:
        file = folder / filename
        st.download_button(
            f"Download {filename}",
            data=file.read_bytes(),
            file_name=filename,
            mime="application/octet-stream",
            on_click="ignore",
            key=f"export_{filename}",
        )
    if st.button("Prepare interactive HTML report"):
        try:
            if "tree" not in locals() or "heatmap" not in locals():
                tree, heatmap = build_figures(
                    folder,
                    cut=cut,
                    search=search or None,
                    max_labels=max_labels,
                    optimal_order=optimal_order,
                )
            tree_html = pio.to_html(tree, full_html=False, include_plotlyjs=True)
            heatmap_html = pio.to_html(heatmap, full_html=False, include_plotlyjs=False)
            page = (
                "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<title>ChoiceMap</title><style>body{font-family:Arial,sans-serif;"
                "max-width:1500px;margin:30px auto;color:#182638}</style></head>"
                "<body><h1>ChoiceMap</h1><p>Customer overlap is not causal "
                "substitution. Branch support comes from customer bootstraps.</p>"
                + tree_html
                + heatmap_html
                + "</body></html>"
            )
            st.session_state.html_export = page.encode("utf-8")
        except Exception as exc:
            st.error(f"Could not prepare HTML export: {exc}")
    if "html_export" in st.session_state:
        st.download_button(
            "Download interactive HTML",
            st.session_state.html_export,
            file_name="choicemap_report.html",
            mime="text/html",
            on_click="ignore",
        )
    st.caption(
        "Exports are held in server memory for download. Avoid uploading private "
        "customer data to an untrusted deployment."
    )
