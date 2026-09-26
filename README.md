# ChoiceMap — transaction-only customer choice trees

ChoiceMap builds an exploratory **behavioral SKU hierarchy** from customer-linked transactions. The engine calculates basket association metrics, customer-repertoire overlap, adjacent category-purchase transitions, and bootstrap branch support. The separate visual layer turns the saved tree into an interactive dendrogram and similarity heatmap.

> **Interpretation matters:** Basket association describes products bought **together**. The choice tree instead clusters SKUs by customers who have bought both over their observed histories. Neither customer overlap nor switching proves that one product would replace another if the other were removed. This is **not** a causal demand-transference model.

## Current status

- `engine.py`: runnable scientific pipeline and command-line interface (schema v3).
- `visual.py`: interactive visual report with 10+ figure factories; reusable `build_figures`, `build_all_figures`, and `write_dashboard` functions.
- `app.py`: Streamlit application with upload, configuration, multi-tab exploration, and cross-filtering.
- Validation and benchmarks on the full yogurt data: **not yet reported**. A small smoke test is not a full-data benchmark.

## Quick start

Use a supported Python environment, preferably Python 3.11 or 3.12, and install dependencies in a virtual environment. The `rtcompat` Polars extra is intended for older CPUs without AVX support; its name is **`rtcompat`**, not `rtcompact` (see [Polars installation](https://docs.pola.rs/user-guide/installation/)).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

Run the supplied yogurt example (after placing the file at `sample_data/instacart_yogurt.parquet`):

```bash
python engine.py --input sample_data/instacart_yogurt.parquet --out output/yogurt
python visual.py --run output/yogurt --out output/yogurt/choice_map.html
streamlit run app.py
```

Open `output/yogurt/choice_map.html` in a browser. The Plotly report is interactive and is saved as standalone HTML; it can be large because it includes Plotly.js ([Plotly HTML export](https://plotly.com/python/interactive-html-export/)).

For a faster first iteration:

```bash
python engine.py --input sample_data/instacart_yogurt.parquet --out output/yogurt_quick --max-skus 40 --bootstrap 5
python visual.py --run output/yogurt_quick --out output/yogurt_quick/choice_map.html --cut 0.65
```

## Expected input

Each row represents a product appearing in an order. The default schema corresponds to the yogurt file:

| Column | Required | Meaning |
| --- | --- | --- |
| `user_id` | Yes | Customer identifier; needed for repertoire overlap and bootstrap |
| `order_id` | Yes | Globally unique basket/order identifier, belonging to exactly one customer |
| `product_id` | Yes | Stable SKU identifier (int or string) |
| `order_number` | Yes by default | Ordering of a customer's orders, used for adjacent category purchase occasions |
| `product_name` | No | Human-readable label; product ID is used when absent |
| `aisle` | Yes under defaults | Category filter; default is `aisle == "yogurt"` |

`user_id`, `order_id`, `product_id`, and—when switching is enabled—`order_number` must be losslessly castable to signed 64-bit integers (or kept as strings). Null required fields and orders assigned to multiple customers are rejected. Repeated `order_id`–`product_id` rows are deduplicated before modeling. Products with fewer than the selected number of distinct buyers are excluded; the top eligible SKUs are selected by buyer count. The report includes customer and order coverage, not just retained SKU count.

`order_dow`, `order_hour_of_day`, `department`, and other source fields are **not used** by the current engine. The sample's `order_hour_of_day` is a string; no conversion is needed because it is not read.

### Other datasets

For a CSV containing exactly one category, disable the category filter:

```bash
python engine.py --input data/transactions.csv --out output/my_category --category-col none --category none
```

The Python `Config` class supports alternative ID/name/sequence/category column names. The current CLI exposes category settings but **does not yet expose CLI flags for renaming ID columns**:

```python
from engine import Config, run

result = run(Config(
    input_path="data/transactions.parquet",
    out="output/my_category",
    user_col="customer_key",
    order_col="basket_key",
    product_col="sku_key",
    name_col="description",
    sequence_col="visit_number",
    category_col=None,
    category_value=None,
))
print(result)
```

If `order_number` is unavailable, pass `--no-switching` to the CLI. Customer-overlap similarity and the dendrogram still work, but sequential transition metrics remain zero.

## Methods

For each pair of retained SKUs, the engine computes:

- **Basket metrics:** joint-order count and support, directional confidence, lift, leverage, and order-level Jaccard.
- **Choice metrics:** shared customer count, customer-level Jaccard, choice distance `1 - customer_jaccard`, overlap lift, and right-tail hypergeometric overlap p-values with Benjamini–Hochberg adjustment.
- **Sequential diagnostics:** weighted directed transitions between consecutive *observed category purchase occasions*. When both occasions contain several SKUs, each cross-pair gets weight `1 / (number of SKUs in first occasion × number in second occasion)`. Transitions are **not** the tree distance.
- **Tree:** exact customer-Jaccard distances, hierarchical clustering (`average` by default; `complete` and `single` available), and cophenetic correlation when defined.
- **Stability:** repeated customer-level bootstrap; branch support is the fraction of resampled trees containing the same exact set of descendant SKUs. A small branch-support value signals an unstable branch, not a statistical test of substitution.

The buyer-repertoire Jaccard for products A and B is:

```text
shared buyers(A, B) / [buyers(A) + buyers(B) - shared buyers(A, B)]
```

The sparse order×SKU and customer×SKU matrices are used to calculate overlap counts. Polars loads and filters transactions; SciPy sparse multiplication handles pair counts and SciPy hierarchical clustering creates the linkage. Polars 2.0's lazy `collect` defaults to its streaming engine, though individual queries and memory use still need empirical measurement ([Polars 2.0 pre-release](https://pola.rs/posts/announcing-polars-2/)).

**Caveat on inference:** The hypergeometric test assumes an independent, fixed-margin null. Repeated purchase histories, exposure, promotions, and varying customer activity complicate that assumption. Treat q-values as exploratory pair annotations, not proof that two SKUs substitute for one another. The engine does not currently perform temporal validation or a consensus tree.

## Outputs

A successful engine run writes files in the `--out` directory:

| File | Contents |
| --- | --- |
| `sku_metrics.parquet` | Retained SKUs, labels, buyer counts, order counts, penetration, **retention/exit rates** |
| `pair_metrics.parquet` | One row per unordered SKU pair with basket, customer, switching, and **expansion rates** |
| `tree_nodes.parquet` | SciPy linkage children, merge height, size, and bootstrap support |
| `cluster_assignments.parquet` | Cluster membership for each SKU at multiple K |
| `cluster_profiles.parquet` | Cluster-level summary statistics |
| `run.json` | Configuration, counts, coverage, leaf order, runtime, and tree-fit diagnostic |

The visual command adds an interactive HTML report at the path given by `--out`. The HTML report includes a color-coded dendrogram, tree-ordered similarity heatmap, branch stability chart, repertoire-basket scatter, SKU behavior map, migration matrix, top migration flows, migration confidence intervals, retention/exit decomposition, and cluster evolution. Faded/dotted branches flag lower bootstrap support, and leaf-marker area helps identify higher-penetration SKUs. Changing the visual cut changes display group colors; it **does not rebuild** the hierarchy. Optional `--optimal-order` rotates equivalent branches for display but may take substantially longer ([SciPy optimal leaf ordering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html)).

For use inside the Streamlit app or custom code:

```python
from visual import build_figures, load_run

data = load_run("output/yogurt")
tree, heatmap = build_figures("output/yogurt", cut=0.7)
# Also available: _make_branch_stability, _make_repertoire_basket_scatter,
# _make_sku_behavior_map, _make_migration_matrix, _make_migration_rank,
# _make_migration_ci, _make_retention_exit_decomposition, _make_cluster_evolution,
# _make_sku_profile, _make_sku_relationship_rankings
```

## CLI options

```bash
python engine.py --help
python visual.py --help
streamlit run app.py
```

Useful engine flags: `--input`, `--out`, `--category-col`, `--category`, `--min-buyers`, `--max-skus`, `--bootstrap`, `--seed`, `--linkage`, `--no-switching`. Useful visual flags: `--run`, `--out`, `--cut`, `--search`, `--max-labels`, `--optimal-order`.

The engine defaults to `min_buyers=30`, `max_skus=120`, `bootstrap=30`, `seed=42`, `linkage=average`, and a maximum of **5,000,000 category-filtered input rows**. The last limit is configured in Python through `Config(max_input_rows=...)` rather than a current CLI flag. Higher SKU caps increase full pair-table, bootstrap, and HTML rendering costs; start small and benchmark on your machine.

## Visualizations overview

### Structure
- **Customer choice tree** — dendrogram with bootstrap support (opacity/dash encoding)
- **Customer repertoire similarity** — tree-ordered heatmap of customer Jaccard
- **Branch stability** — horizontal bar chart of bootstrap support per merge
- **Cluster evolution** — Sankey diagram showing how clusters split across K=4,6,8

### Relationships
- **Repertoire vs basket affinity** — scatter of customer Jaccard vs basket lift (bubble size = shared buyers, color = co-cluster support)
- **SKU behavior map** — penetration vs retention rate (bubble size = exit opportunities)

### Sequential
- **Migration matrix** — heatmap of exit→new-selection rates (rows=source, cols=destination)
- **Top migration flows** — ranked horizontal bars with event counts and rates
- **Migration confidence intervals** — error-bar chart for top flows
- **Retention / exit decomposition** — stacked bars per SKU (retained, exit→new selected, exit→new unselected, exit→no new)

### SKU Explorer (Streamlit)
- **SKU profile** — KPI cards, exit decomposition pie, transition funnel
- **SKU relationship rankings** — top repertoire, basket, migration, expansion partners

## Limitations and next steps

- The current engine collects the filtered data into memory and builds exact SKU-pair tables. It is **not** yet a fully streaming, out-of-core implementation.
- Sequential transitions currently iterate grouped baskets in Python, which may be the first bottleneck on the full yogurt sample; profile this stage before claiming the pipeline is fast.
- A shared customer repertoire can reflect popularity, variety seeking, or exposure. The input contains no price, promotions, availability, or stockout evidence, so assortment removal effects cannot be estimated.
- The visual layer reads the entire pair table; very large SKU universes require a bounded display or alternative rendering strategy.
- Planned: property tests for sparse counts, repeatable full-sample benchmarks, temporal splits, consensus clustering.

## Reproducibility

Record the input-file hash, resolved dependency versions, machine specifications, configuration, elapsed time by stage, peak memory, retained SKU count, and customer coverage when publishing portfolio results. Current `run.json` records only some of these; the rest are development targets. Keep private customer data out of Git and avoid sharing customer-level transaction files in exported demos.