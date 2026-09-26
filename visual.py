"""Interactive visual report for ChoiceMap engine outputs.

Install:
    pip install plotly polars numpy scipy

Run:
    python visual.py --run output/yogurt --out output/yogurt/choice_map.html

For Streamlit:
    from visual import build_figures
    tree, heatmap = build_figures("output/yogurt", cut=0.7)
    st.plotly_chart(tree, use_container_width=True)
    st.plotly_chart(heatmap, use_container_width=True)

This module displays an existing model. It does not refit the tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from scipy.cluster.hierarchy import (
    dendrogram,
    fcluster,
    is_valid_linkage,
    optimal_leaf_ordering,
)
from scipy.spatial.distance import squareform

BACKGROUND = "#F5F7FB"
INK = "#182638"
MUTED = "#63758B"
NEUTRAL = "#9AA8BA"

COLORS = [
    "#3273DC",
    "#E48B36",
    "#009C86",
    "#9B62CC",
    "#D85E72",
    "#69833B",
    "#2B9AB7",
    "#B36B49",
]


def _short(text: str, limit: int = 42) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _color(cluster: int) -> str:
    return COLORS[(cluster - 1) % len(COLORS)]


def load_run(directory: str | Path) -> dict:
    """Load and validate the artifacts produced by engine.py."""
    directory = Path(directory)

    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    sku = pl.read_parquet(directory / "sku_metrics.parquet").sort("product_id")
    pairs = pl.read_parquet(directory / "pair_metrics.parquet")
    nodes = pl.read_parquet(directory / "tree_nodes.parquet").sort("node_id")

    ids = sku["product_id"].to_numpy()
    n = len(ids)

    if n < 2:
        raise ValueError("The run contains fewer than two SKUs.")

    if nodes.height != n - 1:
        raise ValueError("Tree-node count does not match SKU count.")

    if pairs.height != n * (n - 1) // 2:
        raise ValueError("Pair table is incomplete.")

    expected_nodes = np.arange(n, 2 * n - 1)
    if not np.array_equal(nodes["node_id"].to_numpy(), expected_nodes):
        raise ValueError("Tree-node IDs are not aligned with SKU positions.")

    linkage = np.column_stack(
        [
            nodes["left"].to_numpy(),
            nodes["right"].to_numpy(),
            nodes["height"].to_numpy(),
            nodes["leaf_count"].to_numpy(),
        ]
    ).astype(float)

    is_valid_linkage(linkage, throw=True)

    names = [
        str(name or product_id)
        for name, product_id in zip(sku["product_name"].to_list(), ids, strict=True)
    ]

    positions = {int(product_id): i for i, product_id in enumerate(ids)}

    similarity = np.eye(n, dtype=float)
    order_similarity = np.eye(n, dtype=float)
    shared_buyers = np.zeros((n, n), dtype=float)
    basket_lift = np.ones((n, n), dtype=float)

    fields = [
        "product_id_a",
        "product_id_b",
        "customer_jaccard",
        "order_jaccard",
        "shared_customers",
        "basket_lift",
    ]

    for row in pairs.select(fields).iter_rows(named=True):
        a = positions[int(row["product_id_a"])]
        b = positions[int(row["product_id_b"])]

        for matrix, field in (
            (similarity, "customer_jaccard"),
            (order_similarity, "order_jaccard"),
            (shared_buyers, "shared_customers"),
            (basket_lift, "basket_lift"),
        ):
            matrix[a, b] = matrix[b, a] = float(row[field])

    if not np.isfinite(similarity).all():
        raise ValueError("Customer similarity contains non-finite values.")

    if (similarity < 0).any() or (similarity > 1).any():
        raise ValueError("Customer Jaccard must lie between zero and one.")

    return {
        "report": report,
        "sku": sku,
        "nodes": nodes,
        "ids": ids,
        "names": names,
        "linkage": linkage,
        "similarity": similarity,
        "order_similarity": order_similarity,
        "shared_buyers": shared_buyers,
        "basket_lift": basket_lift,
    }


def _geometry(linkage: np.ndarray, leaf_order: list[int]) -> tuple[dict, dict]:
    """Calculate tree coordinates without depending on Matplotlib."""
    n = len(leaf_order)

    locations = {leaf: (0.0, float(position)) for position, leaf in enumerate(leaf_order)}
    descendants = {leaf: frozenset((leaf,)) for leaf in leaf_order}

    for index, (left, right, height, _) in enumerate(linkage):
        left = int(left)
        right = int(right)
        node = n + index

        locations[node] = (
            float(height),
            (locations[left][1] + locations[right][1]) / 2,
        )
        descendants[node] = descendants[left] | descendants[right]

    return locations, descendants


def _layout(title: str, height: int) -> dict:
    return {
        "title": {
            "text": title,
            "x": 0.02,
            "font": {"size": 22, "color": INK},
        },
        "height": height,
        "paper_bgcolor": BACKGROUND,
        "plot_bgcolor": "#FFFFFF",
        "font": {"family": "Arial, sans-serif", "color": INK},
        "hoverlabel": {"bgcolor": "#FFFFFF"},
        "showlegend": False,
    }


def _make_tree(
    data: dict,
    linkage: np.ndarray,
    order: list[int],
    cut: float,
    search: str | None,
    max_labels: int,
) -> go.Figure:
    n = len(order)
    clusters = fcluster(linkage, t=cut, criterion="distance")
    locations, descendants = _geometry(linkage, order)

    supports = data["nodes"]["bootstrap_support"].to_numpy()
    buyers = data["sku"]["buyers"].to_numpy()
    max_buyers = max(int(buyers.max()), 1)

    figure = go.Figure()

    for index, (left, right, _height, _) in enumerate(linkage):
        node = n + index
        left = int(left)
        right = int(right)

        x, _ = locations[node]
        left_x, left_y = locations[left]
        right_x, right_y = locations[right]

        member_clusters = {int(clusters[leaf]) for leaf in descendants[node]}

        single_group = len(member_clusters) == 1
        branch_color = _color(next(iter(member_clusters))) if single_group else NEUTRAL

        support = float(supports[index])
        measured = np.isfinite(support)

        opacity = 0.3 + 0.7 * support if measured else 0.75
        dash = "dot" if measured and support < 0.75 else "solid"

        # Three contiguous segments form a horizontal dendrogram branch.
        figure.add_trace(
            go.Scatter(
                x=[left_x, x, x, right_x],
                y=[left_y, left_y, right_y, right_y],
                mode="lines",
                line={
                    "color": branch_color,
                    "width": 2.5 if single_group else 1.6,
                    "dash": dash,
                },
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        support_text = f"{support:.0%}" if measured else "not calculated"

        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[(left_y + right_y) / 2],
                mode="markers",
                marker={"size": 14, "color": branch_color, "opacity": 0.08},
                customdata=[[node, len(descendants[node]), support_text]],
                hovertemplate=(
                    "<b>Branch %{customdata[0]}</b>"
                    "<br>Merge distance: %{x:.3f}"
                    "<br>SKUs below: %{customdata[1]}"
                    "<br>Bootstrap support: %{customdata[2]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    positions = {leaf: position for position, leaf in enumerate(order)}
    query = search.lower() if search else ""

    for cluster in np.unique(clusters):
        leaves = [leaf for leaf in order if clusters[leaf] == cluster]

        figure.add_trace(
            go.Scatter(
                x=[0] * len(leaves),
                y=[positions[leaf] for leaf in leaves],
                mode="markers",
                marker={
                    "color": _color(int(cluster)),
                    "size": [7 + 12 * np.sqrt(buyers[leaf] / max_buyers) for leaf in leaves],
                    "symbol": [
                        "diamond"
                        if query
                        and (
                            query in data["names"][leaf].lower() or query in str(data["ids"][leaf])
                        )
                        else "circle"
                        for leaf in leaves
                    ],
                    "line": {"color": "white", "width": 1.3},
                },
                customdata=[
                    [
                        data["names"][leaf],
                        int(data["ids"][leaf]),
                        int(buyers[leaf]),
                        float(data["sku"]["buyer_penetration"][leaf]),
                        int(cluster),
                    ]
                    for leaf in leaves
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Product ID: %{customdata[1]}"
                    "<br>Unique buyers: %{customdata[2]:,}"
                    "<br>Buyer penetration: %{customdata[3]:.2%}"
                    "<br>Group: %{customdata[4]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    label_step = max(1, int(np.ceil(n / max_labels)))
    tick_positions = []
    tick_labels = []

    for position, leaf in enumerate(order):
        matches = query and (
            query in data["names"][leaf].lower() or query in str(data["ids"][leaf])
        )

        if position % label_step == 0 or matches:
            tick_positions.append(position)
            tick_labels.append(_short(data["names"][leaf]))

    figure.update_layout(
        **_layout(
            "Customer choice tree · shared buyers, not proven substitution",
            max(630, min(2400, 145 + 22 * n)),
        ),
        margin={"l": 280, "r": 35, "t": 100, "b": 80},
    )

    figure.update_xaxes(
        title="Choice distance: 1 − customer Jaccard",
        range=[
            -0.045,
            min(1.05, float(linkage[-1, 2]) * 1.09 + 0.02),
        ],
        gridcolor="#EDF1F6",
        zeroline=False,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
        range=[n - 0.5, -0.8],
        showgrid=False,
        zeroline=False,
    )

    figure.add_annotation(
        x=1,
        y=1.08,
        xref="paper",
        yref="paper",
        text=(f"{n} SKUs · distance cut {cut:.2f} · {len(np.unique(clusters))} groups"),
        showarrow=False,
        xanchor="right",
        font={"size": 12, "color": MUTED},
    )

    return figure


def _make_heatmap(data: dict, order: list[int]) -> go.Figure:
    n = len(order)
    index = np.ix_(order, order)

    similarity = data["similarity"][index]
    extra = np.stack(
        [
            data["shared_buyers"][index],
            data["order_similarity"][index],
            data["basket_lift"][index],
        ],
        axis=-1,
    )

    figure = go.Figure(
        go.Heatmap(
            z=similarity,
            x=np.arange(n),
            y=np.arange(n),
            customdata=extra,
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#F0F3F9"],
                [0.25, "#D3E5F0"],
                [0.5, "#8FCAC8"],
                [0.75, "#3996A8"],
                [1.0, "#17496F"],
            ],
            colorbar={"title": "Customer<br>Jaccard"},
            hovertemplate=(
                "Row %{y} × column %{x}"
                "<br>Customer Jaccard: %{z:.3f}"
                "<br>Shared buyers: %{customdata[0]:,.0f}"
                "<br>Order Jaccard: %{customdata[1]:.3f}"
                "<br>Basket lift: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    label_step = max(1, int(np.ceil(n / 30)))
    tick_positions = list(range(0, n, label_step))
    tick_labels = [_short(data["names"][order[i]], 34) for i in tick_positions]

    figure.update_layout(
        **_layout(
            "Choice similarity atlas · SKUs ordered by the tree",
            max(680, min(1900, 170 + 16 * n)),
        ),
        margin={"l": 250, "r": 55, "t": 100, "b": 190},
    )

    figure.update_xaxes(
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
        tickangle=-55,
        tickfont={"size": 9},
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
        autorange="reversed",
        tickfont={"size": 9},
        scaleanchor="x",
        scaleratio=1,
    )

    return figure


def build_figures(
    directory: str | Path,
    *,
    cut: float = 0.7,
    search: str | None = None,
    max_labels: int = 55,
    optimal_order: bool = False,
) -> tuple[go.Figure, go.Figure]:
    """Return dendrogram and matching heatmap without refitting the model."""
    if not 0 <= cut <= 1:
        raise ValueError("cut must be between zero and one.")

    if max_labels < 1:
        raise ValueError("max_labels must be positive.")

    data = load_run(directory)
    linkage = data["linkage"]

    if optimal_order:
        distances = squareform(1 - data["similarity"], checks=False)
        linkage = optimal_leaf_ordering(linkage, distances)

    order = [int(leaf) for leaf in dendrogram(linkage, no_plot=True)["leaves"]]

    tree = _make_tree(data, linkage, order, cut, search, max_labels)
    heatmap = _make_heatmap(data, order)

    return tree, heatmap


def write_dashboard(
    directory: str | Path,
    output: str | Path,
    *,
    cut: float = 0.7,
    search: str | None = None,
    max_labels: int = 55,
    optimal_order: bool = False,
) -> Path:
    """Save a standalone, interactive HTML report."""
    data = load_run(directory)

    tree, heatmap = build_figures(
        directory,
        cut=cut,
        search=search,
        max_labels=max_labels,
        optimal_order=optimal_order,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    coverage = float(data["report"]["customer_coverage"])
    quality = data["report"].get("cophenetic_correlation")
    quality_text = "N/A" if quality is None else f"{quality:.2f}"

    cards = [
        ("CATEGORY CUSTOMERS", f"{data['report']['category_customers']:,}"),
        ("SKUS SHOWN", f"{len(data['ids']):,}"),
        ("CUSTOMER COVERAGE", f"{coverage:.1%}"),
        ("TREE FIT", quality_text),
    ]

    card_html = "".join(
        (f'<div class="card"><small>{label}</small><strong>{value}</strong></div>')
        for label, value in cards
    )

    tree_html = pio.to_html(
        tree,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    heatmap_html = pio.to_html(
        heatmap,
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChoiceMap | Customer choice intelligence</title>
<style>
body {{
    margin: 0;
    background: {BACKGROUND};
    color: {INK};
    font-family: Arial, sans-serif;
}}
main {{
    max-width: 1650px;
    margin: auto;
    padding: 30px 22px;
}}
h1 {{ margin-bottom: 8px; }}
.subtitle, .note {{
    color: {MUTED};
    line-height: 1.5;
}}
.cards {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 25px 0;
}}
.card {{
    min-width: 175px;
    padding: 18px 23px;
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 12px;
}}
.card small {{
    display: block;
    color: {MUTED};
    letter-spacing: 1px;
    font-size: 10px;
    margin-bottom: 9px;
}}
.card strong {{ font-size: 25px; }}
section {{
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 14px;
    overflow: hidden;
    margin: 17px 0;
}}
</style>
</head>
<body>
<main>
<h1>ChoiceMap / customer choice intelligence</h1>
<p class="subtitle">
Transaction-only behavioral hierarchy. Colored branches show groups at
distance {cut:.2f}. Faded, dotted branches have weaker customer-bootstrap
support. Larger leaf markers indicate more buyers.
</p>
<div class="cards">{card_html}</div>
<section>{tree_html}</section>
<section>{heatmap_html}</section>
<p class="note">
Shared customers suggest a common repertoire, not proven substitution.
Same-order association is a separate basket signal. Branch support comes
from whole-customer resampling.
</p>
</main>
</body>
</html>"""

    output.write_text(page, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize an existing ChoiceMap engine run.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="output/choice_map.html")
    parser.add_argument("--cut", type=float, default=0.7)
    parser.add_argument("--search")
    parser.add_argument("--max-labels", type=int, default=55)
    parser.add_argument("--optimal-order", action="store_true")

    args = parser.parse_args()

    path = write_dashboard(
        args.run,
        args.out,
        cut=args.cut,
        search=args.search,
        max_labels=args.max_labels,
        optimal_order=args.optimal_order,
    )
    print(path)


if __name__ == "__main__":
    main()
