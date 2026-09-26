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

    # Optional cluster files
    cluster_assignments = None
    cluster_profiles = None
    if (directory / "cluster_assignments.parquet").exists():
        cluster_assignments = pl.read_parquet(directory / "cluster_assignments.parquet")
    if (directory / "cluster_profiles.parquet").exists():
        cluster_profiles = pl.read_parquet(directory / "cluster_profiles.parquet")

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

    positions = {product_id: i for i, product_id in enumerate(ids)}

    similarity = np.eye(n, dtype=float)
    order_similarity = np.eye(n, dtype=float)
    shared_buyers = np.zeros((n, n), dtype=float)
    basket_lift = np.ones((n, n), dtype=float)
    bootstrap_cocluster = np.zeros((n, n), dtype=float)
    customer_overlap_lift = np.ones((n, n), dtype=float)
    expected_shared = np.zeros((n, n), dtype=float)
    excess_shared = np.zeros((n, n), dtype=float)
    migration_a_to_b = np.zeros((n, n), dtype=float)
    migration_b_to_a = np.zeros((n, n), dtype=float)
    migration_rate_a_to_b = np.full((n, n), np.nan, dtype=float)
    migration_rate_b_to_a = np.full((n, n), np.nan, dtype=float)
    migration_ci_lower_a_to_b = np.full((n, n), np.nan, dtype=float)
    migration_ci_upper_a_to_b = np.full((n, n), np.nan, dtype=float)
    migration_ci_lower_b_to_a = np.full((n, n), np.nan, dtype=float)
    migration_ci_upper_b_to_a = np.full((n, n), np.nan, dtype=float)
    migration_reliable_a = np.zeros((n, n), dtype=bool)
    migration_reliable_b = np.zeros((n, n), dtype=bool)
    expansion_a_to_b = np.zeros((n, n), dtype=float)
    expansion_b_to_a = np.zeros((n, n), dtype=float)
    expansion_rate_a_to_b = np.full((n, n), np.nan, dtype=float)
    expansion_rate_b_to_a = np.full((n, n), np.nan, dtype=float)

    # Batch-fill matrices using vectorized operations
    a_idx = np.array([positions[pid] for pid in pairs["product_id_a"].to_list()])
    b_idx = np.array([positions[pid] for pid in pairs["product_id_b"].to_list()])

    matrix_field_pairs = [
        (similarity, "customer_jaccard"),
        (order_similarity, "order_jaccard"),
        (shared_buyers, "shared_customers"),
        (basket_lift, "basket_lift"),
        (bootstrap_cocluster, "bootstrap_pair_cocluster"),
        (customer_overlap_lift, "customer_overlap_lift"),
        (expected_shared, "expected_shared_customers"),
        (excess_shared, "excess_shared_customers"),
        (migration_a_to_b, "migration_a_to_b"),
        (migration_b_to_a, "migration_b_to_a"),
        (migration_rate_a_to_b, "migration_rate_a_to_b"),
        (migration_rate_b_to_a, "migration_rate_b_to_a"),
        (migration_ci_lower_a_to_b, "migration_ci_lower_a_to_b"),
        (migration_ci_upper_a_to_b, "migration_ci_upper_a_to_b"),
        (migration_ci_lower_b_to_a, "migration_ci_lower_b_to_a"),
        (migration_ci_upper_b_to_a, "migration_ci_upper_b_to_a"),
        (migration_reliable_a, "migration_reliable_a"),
        (migration_reliable_b, "migration_reliable_b"),
        (expansion_a_to_b, "expansion_a_to_b"),
        (expansion_b_to_a, "expansion_b_to_a"),
        (expansion_rate_a_to_b, "expansion_rate_a_to_b"),
        (expansion_rate_b_to_a, "expansion_rate_b_to_a"),
    ]

    for matrix, col in matrix_field_pairs:
        vals = pairs[col].to_numpy()
        matrix[a_idx, b_idx] = vals
        matrix[b_idx, a_idx] = vals

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
        "bootstrap_cocluster": bootstrap_cocluster,
        "customer_overlap_lift": customer_overlap_lift,
        "expected_shared": expected_shared,
        "excess_shared": excess_shared,
        "migration_a_to_b": migration_a_to_b,
        "migration_b_to_a": migration_b_to_a,
        "migration_rate_a_to_b": migration_rate_a_to_b,
        "migration_rate_b_to_a": migration_rate_b_to_a,
        "migration_ci_lower_a_to_b": migration_ci_lower_a_to_b,
        "migration_ci_upper_a_to_b": migration_ci_upper_a_to_b,
        "migration_ci_lower_b_to_a": migration_ci_lower_b_to_a,
        "migration_ci_upper_b_to_a": migration_ci_upper_b_to_a,
        "migration_reliable_a": migration_reliable_a,
        "migration_reliable_b": migration_reliable_b,
        "expansion_a_to_b": expansion_a_to_b,
        "expansion_b_to_a": expansion_b_to_a,
        "expansion_rate_a_to_b": expansion_rate_a_to_b,
        "expansion_rate_b_to_a": expansion_rate_b_to_a,
        "cluster_assignments": cluster_assignments,
        "cluster_profiles": cluster_profiles,
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
                        data["ids"][leaf],
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


def _make_branch_stability(data: dict, min_support: float = 0.0) -> go.Figure:
    """Horizontal bar chart of branch bootstrap support."""
    supports = data["nodes"]["bootstrap_support"].to_numpy()
    n = len(data["ids"])
    node_ids = np.arange(n, 2 * n - 1)

    # Filter by minimum support
    mask = np.isfinite(supports) & (supports >= min_support)
    filtered_supports = supports[mask]
    filtered_nodes = node_ids[mask]

    # Sort by support descending
    sort_idx = np.argsort(filtered_supports)[::-1]
    filtered_supports = filtered_supports[sort_idx]
    filtered_nodes = filtered_nodes[sort_idx]

    # Create labels
    labels = [f"Branch {int(node)}" for node in filtered_nodes]
    percentages = [f"{s:.0%}" for s in filtered_supports]

    figure = go.Figure(
        go.Bar(
            x=filtered_supports,
            y=labels,
            orientation="h",
            marker_color="#3273DC",
            text=percentages,
            textposition="outside",
            hovertemplate=("<b>%{y}</b><br>Bootstrap support: %{x:.1%}<extra></extra>"),
        )
    )

    figure.update_layout(
        **_layout(
            "Branch stability · bootstrap support by merge",
            max(400, 100 + 25 * len(filtered_supports)),
        ),
        margin={"l": 120, "r": 80, "t": 80, "b": 60},
        xaxis={"title": "Bootstrap support", "range": [0, 1.05], "gridcolor": "#EDF1F6"},
        yaxis={"autorange": "reversed"},
    )

    figure.add_vline(x=0.75, line_dash="dash", line_color=MUTED, opacity=0.5)
    figure.add_annotation(
        x=0.75,
        y=1.02,
        xref="x",
        yref="paper",
        text="75% threshold",
        showarrow=False,
        font={"size": 10, "color": MUTED},
        xanchor="left",
    )

    return figure


def _make_repertoire_basket_scatter(
    data: dict, min_shared: int = 1, min_cocluster: float = 0.0
) -> go.Figure:
    """Repertoire vs basket affinity scatter plot."""
    n = len(data["ids"])
    a, b = np.triu_indices(n, 1)

    x = data["similarity"][a, b]  # customer_jaccard
    y = data["basket_lift"][a, b]  # basket_lift
    sizes = data["shared_buyers"][a, b]
    cocluster = data["bootstrap_cocluster"][a, b]
    reliable = data["migration_reliable_a"][a, b] | data["migration_reliable_b"][a, b]

    # Filter
    mask = (sizes >= min_shared) & (cocluster >= min_cocluster) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    sizes = sizes[mask]
    cocluster = cocluster[mask]
    reliable = reliable[mask]
    pair_a = a[mask]
    pair_b = b[mask]

    # Normalize sizes for marker
    max_size = max(sizes.max(), 1) if len(sizes) > 0 else 1
    marker_sizes = 8 + 20 * np.sqrt(sizes / max_size)

    # Color by co-cluster support
    colors = np.where(
        np.isfinite(cocluster),
        np.clip(cocluster * 255, 0, 255).astype(int),
        128,
    )

    figure = go.Figure()

    # Add quadrant lines
    x_median = np.median(x) if len(x) > 0 else 0.5
    y_median = np.median(y) if len(y) > 0 else 1.0

    figure.add_vline(x=x_median, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
    figure.add_hline(y=y_median, line_dash="dot", line_color=NEUTRAL, opacity=0.5)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": colors,
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Bootstrap<br>co-cluster"},
                "cmin": 0,
                "cmax": 1,
                "opacity": 0.7,
                "line": {"color": "white", "width": 0.5},
            },
            customdata=np.column_stack(
                [
                    [data["names"][i] for i in pair_a],
                    [data["names"][i] for i in pair_b],
                    [data["ids"][i] for i in pair_a],
                    [data["ids"][i] for i in pair_b],
                    sizes,
                    x,
                    y,
                    cocluster,
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b> ↔ <b>%{customdata[1]}</b>"
                "<br>Product IDs: %{customdata[2]} × %{customdata[3]}"
                "<br>Shared buyers: %{customdata[4]:,.0f}"
                "<br>Customer Jaccard (repertoire): %{customdata[5]:.3f}"
                "<br>Basket lift: %{customdata[6]:.2f}x"
                "<br>Bootstrap co-cluster: %{customdata[7]:.1%}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    figure.update_layout(
        **_layout("Repertoire vs basket affinity", 700),
        margin={"l": 80, "r": 80, "t": 80, "b": 80},
        xaxis={
            "title": "Customer Jaccard (repertoire overlap)",
            "range": [-0.02, 1.02],
            "gridcolor": "#EDF1F6",
        },
        yaxis={
            "title": "Basket lift (same-order association)",
            "range": [0, max(y.max() * 1.1, 1.1) if len(y) > 0 else 2],
            "gridcolor": "#EDF1F6",
        },
    )

    # Add quadrant annotations
    if len(x) > 0:
        figure.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text="High repertoire<br>High basket",
            showarrow=False,
            font={"size": 10, "color": MUTED},
            xanchor="right",
            yanchor="top",
        )
        figure.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text="Low repertoire<br>High basket",
            showarrow=False,
            font={"size": 10, "color": MUTED},
            xanchor="left",
            yanchor="top",
        )
        figure.add_annotation(
            x=0.95,
            y=0.05,
            xref="paper",
            yref="paper",
            text="High repertoire<br>Low basket",
            showarrow=False,
            font={"size": 10, "color": MUTED},
            xanchor="right",
            yanchor="bottom",
        )
        figure.add_annotation(
            x=0.05,
            y=0.05,
            xref="paper",
            yref="paper",
            text="Low repertoire<br>Low basket",
            showarrow=False,
            font={"size": 10, "color": MUTED},
            xanchor="left",
            yanchor="bottom",
        )

    return figure


def _make_sku_behavior_map(data: dict) -> go.Figure:
    """SKU behavior map: buyer penetration vs retention rate."""
    sku = data["sku"]
    penetration = sku["buyer_penetration"].to_numpy()
    retention_rate = sku["retention_rate"].to_numpy()
    exit_opps = sku["exit_opportunities"].to_numpy()
    names = data["names"]
    ids = data["ids"]

    # Filter valid values
    mask = np.isfinite(penetration) & np.isfinite(retention_rate) & np.isfinite(exit_opps)
    penetration = penetration[mask]
    retention_rate = retention_rate[mask]
    exit_opps = exit_opps[mask]
    filtered_names = [names[i] for i in np.where(mask)[0]]
    filtered_ids = [ids[i] for i in np.where(mask)[0]]

    # Handle empty data (no switching)
    if len(penetration) == 0:
        figure = go.Figure()
        figure.update_layout(
            **_layout("SKU behavior map · penetration vs retention (no switching data)", 400),
            margin={"l": 80, "r": 80, "t": 80, "b": 80},
            xaxis={"title": "Buyer penetration", "gridcolor": "#EDF1F6"},
            yaxis={"title": "Retention rate", "gridcolor": "#EDF1F6"},
        )
        figure.add_annotation(
            text="No switching data available. Enable order sequence in sidebar.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": MUTED},
        )
        return figure

    max_exits = max(exit_opps.max(), 1)
    marker_sizes = 8 + 25 * np.sqrt(exit_opps / max_exits)

    figure = go.Figure(
        go.Scatter(
            x=penetration,
            y=retention_rate,
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": penetration,
                "colorscale": "Blues",
                "showscale": True,
                "colorbar": {"title": "Buyer<br>penetration"},
                "cmin": 0,
                "cmax": max(penetration.max(), 0.01),
                "opacity": 0.8,
                "line": {"color": "white", "width": 1},
            },
            customdata=np.column_stack(
                [filtered_names, filtered_ids, exit_opps, penetration, retention_rate]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Product ID: %{customdata[1]}"
                "<br>Exit opportunities: %{customdata[2]:,.0f}"
                "<br>Buyer penetration: %{customdata[3]:.2%}"
                "<br>Retention rate: %{customdata[4]:.2%}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Add median lines
    if len(penetration) > 0:
        x_med = np.median(penetration)
        y_med = np.median(retention_rate)
        figure.add_vline(x=x_med, line_dash="dot", line_color=NEUTRAL, opacity=0.5)
        figure.add_hline(y=y_med, line_dash="dot", line_color=NEUTRAL, opacity=0.5)

    figure.update_layout(
        **_layout("SKU behavior map · penetration vs retention", 650),
        margin={"l": 80, "r": 80, "t": 80, "b": 80},
        xaxis={
            "title": "Buyer penetration",
            "range": [-0.02, max(penetration.max() * 1.1, 0.1) if len(penetration) > 0 else 1],
            "gridcolor": "#EDF1F6",
            "tickformat": ".0%",
        },
        yaxis={
            "title": "Retention rate",
            "range": [-0.02, 1.02],
            "gridcolor": "#EDF1F6",
            "tickformat": ".0%",
        },
    )

    return figure


def _make_cluster_evolution(data: dict) -> go.Figure | None:
    """Cluster evolution Sankey: shows how clusters split from K to K+1."""
    assignments = data.get("cluster_assignments")
    if assignments is None:
        return None

    # Find cluster columns (cluster_k4, cluster_k6, cluster_k8, etc.)
    cluster_cols = [c for c in assignments.columns if c.startswith("cluster_k")]
    if len(cluster_cols) < 2:
        return None

    # Sort by K value
    cluster_cols.sort(key=lambda x: int(x.replace("cluster_k", "")))

    # Build Sankey data
    labels = []
    label_to_idx = {}
    sources = []
    targets = []
    values = []

    # Track cluster sizes at each K
    cluster_sizes = {}

    for _k_idx, col in enumerate(cluster_cols):
        k_val = int(col.replace("cluster_k", ""))
        clusters = assignments[col].to_numpy()
        unique_clusters = sorted(np.unique(clusters))

        for cl in unique_clusters:
            label = f"K={k_val}: Cluster {cl}"
            label_to_idx[label] = len(labels)
            labels.append(label)
            mask = clusters == cl
            cluster_sizes[(k_val, cl)] = int(mask.sum())

    # Build flows between consecutive K levels
    for i in range(len(cluster_cols) - 1):
        col_from = cluster_cols[i]
        col_to = cluster_cols[i + 1]
        k_from = int(col_from.replace("cluster_k", ""))
        k_to = int(col_to.replace("cluster_k", ""))

        assignments_from = assignments[col_from].to_numpy()
        assignments_to = assignments[col_to].to_numpy()

        for cl_from in sorted(np.unique(assignments_from)):
            mask_from = assignments_from == cl_from
            downstream = assignments_to[mask_from]
            for cl_to in sorted(np.unique(downstream)):
                count = int((downstream == cl_to).sum())
                if count > 0:
                    label_from = f"K={k_from}: Cluster {cl_from}"
                    label_to = f"K={k_to}: Cluster {cl_to}"
                    sources.append(label_to_idx[label_from])
                    targets.append(label_to_idx[label_to])
                    values.append(count)

    if not sources:
        return None

    # Color by K level
    node_colors = []
    for label in labels:
        k_val = int(label.split(":")[0].replace("K=", ""))
        color_idx = (k_val - 2) % len(COLORS) if k_val >= 2 else 0
        node_colors.append(COLORS[color_idx])

    figure = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "color": node_colors,
                "pad": 20,
                "thickness": 20,
                "line": {"color": "white", "width": 1},
            },
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": "rgba(100,100,100,0.3)",
            },
        )
    )

    figure.update_layout(
        **_layout("Cluster evolution · how groups split across K", 600),
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
    )

    return figure


def _make_migration_matrix(
    data: dict,
    min_reliability: bool = True,
    top_n: int | None = None,
) -> go.Figure:
    """Migration rate matrix: source SKU (rows) → destination SKU (columns)."""
    n = len(data["ids"])
    names = data["names"]
    ids = data["ids"]
    migration_rate = data["migration_rate_a_to_b"]
    reliable_a = data["migration_reliable_a"]

    # Create full matrix (migration_rate_a_to_b is upper triangular)
    matrix = np.full((n, n), np.nan)
    a, b = np.triu_indices(n, 1)
    matrix[a, b] = migration_rate[a, b]
    matrix[b, a] = data["migration_rate_b_to_a"][a, b]

    # Build reliability mask
    reliable = np.zeros((n, n), dtype=bool)
    reliable[a, b] = reliable_a[a, b]
    reliable[b, a] = data["migration_reliable_b"][a, b]

    # Apply reliability filter
    if min_reliability:
        matrix = np.where(reliable, matrix, np.nan)

    # Optionally filter to top N SKUs by exit opportunities
    if top_n is not None:
        exit_opps = data["sku"]["exit_opportunities"].to_numpy()
        top_indices = np.argsort(exit_opps)[::-1][:top_n]
        matrix = matrix[np.ix_(top_indices, top_indices)]
        names = [names[i] for i in top_indices]
        ids = [ids[i] for i in top_indices]

    labels = [f"{names[i]} ({ids[i]})" for i in range(len(names))]

    finite_vals = matrix[np.isfinite(matrix)]
    zmax = float(finite_vals.max()) if len(finite_vals) > 0 else 1.0

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale="RdYlBu_r",
            zmin=0,
            zmax=zmax,
            colorbar={"title": "Migration<br>rate"},
            hovertemplate=(
                "Source: %{y}<br>Destination: %{x}<br>Migration rate: %{z:.2%}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        **_layout("Migration matrix · exit → new selection", 700),
        margin={"l": 200, "r": 50, "t": 100, "b": 200},
    )

    figure.update_xaxes(tickangle=-55, tickfont={"size": 9})
    figure.update_yaxes(tickfont={"size": 9})

    return figure


def _make_migration_rank(data: dict, top_n: int = 20) -> go.Figure:
    """Top observed migration flows as horizontal bar chart."""
    n = len(data["ids"])
    a, b = np.triu_indices(n, 1)

    # Combine both directions
    migration_a = data["migration_a_to_b"][a, b]
    migration_b = data["migration_b_to_a"][a, b]
    rate_a = data["migration_rate_a_to_b"][a, b]
    rate_b = data["migration_rate_b_to_a"][a, b]
    reliable_a = data["migration_reliable_a"][a, b]
    reliable_b = data["migration_reliable_b"][a, b]

    # Build combined arrays
    pairs = []
    for i in range(len(a)):
        if reliable_a[i] and migration_a[i] > 0:
            pairs.append(
                (
                    data["names"][a[i]],
                    data["names"][b[i]],
                    data["ids"][a[i]],
                    data["ids"][b[i]],
                    int(migration_a[i]),
                    float(rate_a[i]),
                    "a→b",
                )
            )
        if reliable_b[i] and migration_b[i] > 0:
            pairs.append(
                (
                    data["names"][b[i]],
                    data["names"][a[i]],
                    data["ids"][b[i]],
                    data["ids"][a[i]],
                    int(migration_b[i]),
                    float(rate_b[i]),
                    "b→a",
                )
            )

    if not pairs:
        figure = go.Figure()
        figure.update_layout(**_layout("Top migration flows · none reliable", 400))
        return figure

    # Sort by count descending
    pairs.sort(key=lambda x: x[4], reverse=True)
    pairs = pairs[:top_n]

    sources = [p[0] for p in pairs]
    dests = [p[1] for p in pairs]
    counts = [p[4] for p in pairs]
    rates = [p[5] for p in pairs]

    labels = [f"{s} → {d}" for s, d in zip(sources, dests, strict=True)]

    figure = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color="#E48B36",
            text=[f"{c:,} events · {r:.1%}" for c, r in zip(counts, rates, strict=True)],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Events: %{x:,}<br>Migration rate: %{customdata:.1%}<extra></extra>"
            ),
            customdata=rates,
        )
    )

    figure.update_layout(
        **_layout(f"Top {len(pairs)} observed migration flows", max(400, 100 + 25 * len(pairs))),
        margin={"l": 280, "r": 80, "t": 80, "b": 60},
        xaxis={"title": "Migration events", "gridcolor": "#EDF1F6"},
        yaxis={"autorange": "reversed"},
    )

    return figure


def _make_migration_ci(data: dict, top_n: int = 15) -> go.Figure:
    """Migration confidence intervals for top flows."""
    n = len(data["ids"])
    a, b = np.triu_indices(n, 1)

    migration_a = data["migration_a_to_b"][a, b]
    migration_b = data["migration_b_to_a"][a, b]
    rate_a = data["migration_rate_a_to_b"][a, b]
    rate_b = data["migration_rate_b_to_a"][a, b]
    ci_lower_a = data["migration_ci_lower_a_to_b"][a, b]
    ci_upper_a = data["migration_ci_upper_a_to_b"][a, b]
    ci_lower_b = data["migration_ci_lower_b_to_a"][a, b]
    ci_upper_b = data["migration_ci_upper_b_to_a"][a, b]
    reliable_a = data["migration_reliable_a"][a, b]
    reliable_b = data["migration_reliable_b"][a, b]

    pairs = []
    for i in range(len(a)):
        if reliable_a[i] and migration_a[i] > 0 and np.isfinite(rate_a[i]):
            pairs.append(
                (
                    data["names"][a[i]],
                    data["names"][b[i]],
                    data["ids"][a[i]],
                    data["ids"][b[i]],
                    float(rate_a[i]),
                    float(ci_lower_a[i]),
                    float(ci_upper_a[i]),
                )
            )
        if reliable_b[i] and migration_b[i] > 0 and np.isfinite(rate_b[i]):
            pairs.append(
                (
                    data["names"][b[i]],
                    data["names"][a[i]],
                    data["ids"][b[i]],
                    data["ids"][a[i]],
                    float(rate_b[i]),
                    float(ci_lower_b[i]),
                    float(ci_upper_b[i]),
                )
            )

    if not pairs:
        figure = go.Figure()
        figure.update_layout(**_layout("Migration confidence intervals · none reliable", 400))
        return figure

    # Sort by rate descending
    pairs.sort(key=lambda x: x[4], reverse=True)
    pairs = pairs[:top_n]

    labels = [f"{p[0]} → {p[1]}" for p in pairs]
    rates = [p[4] for p in pairs]
    lowers = [p[5] for p in pairs]
    uppers = [p[6] for p in pairs]

    figure = go.Figure()

    # Add error bars
    figure.add_trace(
        go.Scatter(
            x=rates,
            y=labels,
            mode="markers",
            marker={"size": 12, "color": "#3273DC"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [u - r for r, u in zip(rates, uppers, strict=True)],
                "arrayminus": [r - low for r, low in zip(rates, lowers, strict=True)],
                "color": MUTED,
                "thickness": 2,
                "width": 4,
            },
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Migration rate: %{x:.1%}"
                "<br>95% CI: [%{customdata[0]:.1%}, %{customdata[1]:.1%}]"
                "<extra></extra>"
            ),
            customdata=list(zip(lowers, uppers, strict=True)),
            showlegend=False,
        )
    )

    figure.update_layout(
        **_layout(
            f"Migration confidence intervals · top {len(pairs)} flows",
            max(400, 100 + 25 * len(pairs)),
        ),
        margin={"l": 280, "r": 80, "t": 80, "b": 60},
        xaxis={
            "title": "Migration rate",
            "range": [0, max(uppers) * 1.1 if uppers else 0.5],
            "gridcolor": "#EDF1F6",
            "tickformat": ".0%",
        },
        yaxis={"autorange": "reversed"},
    )

    return figure


def _make_retention_exit_decomposition(data: dict, top_n: int = 15) -> go.Figure:
    """Stacked bar: retention / exit_new_selected / exit_new_unselected / exit_no_new."""
    sku = data["sku"]

    # Sort by exit opportunities descending
    exit_opps = sku["exit_opportunities"].to_numpy()
    order = np.argsort(exit_opps)[::-1][:top_n]

    names = [data["names"][i] for i in order]
    ids = [data["ids"][i] for i in order]

    retained = sku["retention_occasions"].to_numpy()[order]
    exit_new_sel = sku["exit_new_selected"].to_numpy()[order]
    exit_new_unsel = sku["exit_new_unselected_only"].to_numpy()[order]
    exit_no_new = sku["exit_no_new_sku"].to_numpy()[order]
    total = retained + exit_new_sel + exit_new_unsel + exit_no_new

    # Convert to percentages
    p_retained = np.divide(
        retained, total, out=np.zeros_like(retained, dtype=float), where=total > 0
    )
    p_new_sel = np.divide(
        exit_new_sel, total, out=np.zeros_like(exit_new_sel, dtype=float), where=total > 0
    )
    p_new_unsel = np.divide(
        exit_new_unsel, total, out=np.zeros_like(exit_new_unsel, dtype=float), where=total > 0
    )
    p_no_new = np.divide(
        exit_no_new, total, out=np.zeros_like(exit_no_new, dtype=float), where=total > 0
    )

    labels = [f"{n} ({i})" for n, i in zip(names, ids, strict=True)]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            y=labels,
            x=p_retained,
            orientation="h",
            name="Retained",
            marker_color="#009C86",
            hovertemplate="Retained: %{x:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=labels,
            x=p_new_sel,
            orientation="h",
            name="Exit → new selected",
            marker_color="#3273DC",
            hovertemplate="Exit → new selected: %{x:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=labels,
            x=p_new_unsel,
            orientation="h",
            name="Exit → new unselected only",
            marker_color="#E48B36",
            hovertemplate="Exit → new unselected only: %{x:.1%}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            y=labels,
            x=p_no_new,
            orientation="h",
            name="Exit → no new SKU",
            marker_color="#D85E72",
            hovertemplate="Exit → no new SKU: %{x:.1%}<extra></extra>",
        )
    )

    figure.update_layout(
        **_layout(
            f"Retention / exit decomposition · top {len(labels)} SKUs",
            max(400, 100 + 25 * len(labels)),
        ),
        margin={"l": 200, "r": 80, "t": 80, "b": 60},
        barmode="stack",
        xaxis={
            "title": "Proportion of category occasions",
            "range": [0, 1.05],
            "gridcolor": "#EDF1F6",
            "tickformat": ".0%",
        },
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": -0.15, "x": 0.5, "xanchor": "center"},
    )

    return figure


def _make_transition_funnel(data: dict, sku_index: int) -> go.Figure:
    """Transition opportunity funnel for a single SKU."""
    sku = data["sku"]

    trans_opps = int(sku["transition_opportunities"].to_numpy()[sku_index])
    retained = int(sku["retention_occasions"].to_numpy()[sku_index])
    exit_opps = int(sku["exit_opportunities"].to_numpy()[sku_index])
    exit_new_sel = int(sku["exit_new_selected"].to_numpy()[sku_index])
    exit_new_unsel = int(sku["exit_new_unselected_only"].to_numpy()[sku_index])
    exit_no_new = int(sku["exit_no_new_sku"].to_numpy()[sku_index])

    name = data["names"][sku_index]
    sku_id = data["ids"][sku_index]

    stages = [
        ("Transition opportunities", trans_opps),
        ("Retained", retained),
        ("Exited", exit_opps),
        ("Exit → new selected", exit_new_sel),
        ("Exit → new unselected", exit_new_unsel),
        ("Exit → no new SKU", exit_no_new),
    ]

    figure = go.Figure(
        go.Funnel(
            y=[s[0] for s in stages],
            x=[s[1] for s in stages],
            textinfo="value+percent initial",
            marker={"color": ["#3273DC", "#009C86", "#D85E72", "#E48B36", "#9B62CC", "#69833B"]},
            textfont={"size": 14},
        )
    )

    figure.update_layout(
        **_layout(f"Transition funnel · {name} ({sku_id})", 500),
        margin={"l": 200, "r": 80, "t": 80, "b": 60},
    )

    return figure


def _make_sku_profile(data: dict, sku_index: int) -> go.Figure:
    """Detailed profile for a single SKU."""
    sku = data["sku"]
    name = data["names"][sku_index]
    sku_id = data["ids"][sku_index]

    buyers = int(sku["buyers"].to_numpy()[sku_index])
    cat_orders = int(sku["category_orders"].to_numpy()[sku_index])
    penetration = float(sku["buyer_penetration"].to_numpy()[sku_index])
    order_support = float(sku["order_support"].to_numpy()[sku_index])
    retention_rate = float(sku["retention_rate"].to_numpy()[sku_index])
    exit_rate = float(sku["exit_rate"].to_numpy()[sku_index])
    trans_opps = int(sku["transition_opportunities"].to_numpy()[sku_index])
    exit_opps = int(sku["exit_opportunities"].to_numpy()[sku_index])
    retained = int(sku["retention_occasions"].to_numpy()[sku_index])
    exit_new_sel = int(sku["exit_new_selected"].to_numpy()[sku_index])
    exit_new_unsel = int(sku["exit_new_unselected_only"].to_numpy()[sku_index])
    exit_no_new = int(sku["exit_no_new_sku"].to_numpy()[sku_index])

    figure = go.Figure()

    # KPI row
    kpis = [
        ("Buyers", f"{buyers:,}"),
        ("Category orders", f"{cat_orders:,}"),
        ("Buyer penetration", f"{penetration:.1%}"),
        ("Order support", f"{order_support:.1%}"),
        ("Retention rate", f"{retention_rate:.1%}"),
        ("Exit rate", f"{exit_rate:.1%}"),
    ]

    for i, (label, value) in enumerate(kpis):
        figure.add_annotation(
            x=0.05 + (i % 3) * 0.32,
            y=0.95 - (i // 3) * 0.25,
            xref="paper",
            yref="paper",
            text=f"<b>{label}</b><br><span style='font-size:24px'>{value}</span>",
            showarrow=False,
            align="center",
            bgcolor="white",
            bordercolor="#E5EAF2",
            borderwidth=1,
            borderpad=10,
        )

    # Exit decomposition pie
    exit_labels = ["Retained", "Exit → new selected", "Exit → new unselected", "Exit → no new"]
    exit_values = [retained, exit_new_sel, exit_new_unsel, exit_no_new]
    exit_colors = ["#009C86", "#3273DC", "#E48B36", "#D85E72"]

    figure.add_trace(
        go.Pie(
            labels=exit_labels,
            values=exit_values,
            domain={"x": [0.02, 0.48], "y": [0.02, 0.45]},
            marker={"colors": exit_colors},
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            showlegend=False,
        )
    )

    figure.add_annotation(
        x=0.25,
        y=0.48,
        xref="paper",
        yref="paper",
        text="Exit decomposition",
        showarrow=False,
        font={"size": 14, "color": INK},
        xanchor="center",
    )

    # Transition funnel (using bar chart instead of funnel for domain compatibility)
    stages = [
        ("Transition opps", trans_opps),
        ("Retained", retained),
        ("Exited", exit_opps),
        ("→ new selected", exit_new_sel),
        ("→ new unselected", exit_new_unsel),
        ("→ no new", exit_no_new),
    ]

    figure.add_trace(
        go.Bar(
            y=[s[0] for s in stages],
            x=[s[1] for s in stages],
            orientation="h",
            marker={"color": ["#3273DC", "#009C86", "#D85E72", "#E48B36", "#9B62CC", "#69833B"]},
            text=[str(s[1]) for s in stages],
            textposition="auto",
            textfont={"size": 11},
            hovertemplate="%{y}: %{x:,}<extra></extra>",
            showlegend=False,
        )
    )

    figure.update_layout(
        **_layout(f"SKU profile · {name} ({sku_id})", 500),
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
        barmode="overlay",
    )

    return figure


def _make_sku_relationship_rankings(data: dict, sku_index: int, top_n: int = 10) -> go.Figure:
    """Top relationships for a selected SKU across repertoire, basket, migration, expansion."""
    names = data["names"]
    ids = data["ids"]

    # Repertoire (customer Jaccard)
    repertoire_scores = data["similarity"][sku_index, :].copy()
    repertoire_scores[sku_index] = -1  # exclude self
    rep_order = np.argsort(repertoire_scores)[::-1][:top_n]
    rep_names = [f"{names[i]} ({ids[i]})" for i in rep_order]
    rep_values = repertoire_scores[rep_order]

    # Basket (basket lift)
    basket_scores = data["basket_lift"][sku_index, :].copy()
    basket_scores[sku_index] = -1
    basket_order = np.argsort(basket_scores)[::-1][:top_n]
    basket_names = [f"{names[i]} ({ids[i]})" for i in basket_order]
    basket_values = basket_scores[basket_order]

    # Migration (outbound rate)
    mig_out = data["migration_rate_a_to_b"][sku_index, :].copy()
    mig_out[sku_index] = np.nan
    # Also check reverse direction
    mig_out_rev = data["migration_rate_b_to_a"][:, sku_index].copy()
    mig_out_rev[sku_index] = np.nan
    # Combine: for each other SKU, take the max rate in either direction
    mig_combined = np.maximum(mig_out, mig_out_rev)
    mig_order = np.argsort(np.nan_to_num(mig_combined, nan=-1))[::-1][:top_n]
    mig_names = [f"{names[i]} ({ids[i]})" for i in mig_order]
    mig_values = mig_combined[mig_order]

    # Expansion (outbound rate)
    exp_out = data["expansion_rate_a_to_b"][sku_index, :].copy()
    exp_out[sku_index] = np.nan
    exp_out_rev = data["expansion_rate_b_to_a"][:, sku_index].copy()
    exp_out_rev[sku_index] = np.nan
    exp_combined = np.maximum(exp_out, exp_out_rev)
    exp_order = np.argsort(np.nan_to_num(exp_combined, nan=-1))[::-1][:top_n]
    exp_names = [f"{names[i]} ({ids[i]})" for i in exp_order]
    exp_values = exp_combined[exp_order]

    figure = go.Figure()

    # Repertoire
    figure.add_trace(
        go.Bar(
            y=rep_names,
            x=rep_values,
            orientation="h",
            name="Repertoire (Jaccard)",
            marker_color="#3273DC",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    # Basket
    figure.add_trace(
        go.Bar(
            y=basket_names,
            x=basket_values,
            orientation="h",
            name="Basket (lift)",
            marker_color="#009C86",
            hovertemplate="%{y}: %{x:.2f}x<extra></extra>",
        )
    )
    # Migration
    figure.add_trace(
        go.Bar(
            y=mig_names,
            x=mig_values,
            orientation="h",
            name="Migration (rate)",
            marker_color="#E48B36",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )
    # Expansion
    figure.add_trace(
        go.Bar(
            y=exp_names,
            x=exp_values,
            orientation="h",
            name="Expansion (rate)",
            marker_color="#9B62CC",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
        )
    )

    figure.update_layout(
        **_layout(
            f"SKU relationships · {data['names'][sku_index]} ({data['ids'][sku_index]})",
            max(500, 150 + 30 * top_n),
        ),
        margin={"l": 250, "r": 80, "t": 80, "b": 60},
        barmode="overlay",
        xaxis={"title": "Score", "gridcolor": "#EDF1F6"},
        yaxis={"autorange": "reversed"},
        legend={"orientation": "h", "y": -0.15, "x": 0.5, "xanchor": "center"},
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
    """Save a standalone, interactive HTML report with all visualizations."""
    data = load_run(directory)

    # Build all figures
    tree, heatmap = build_figures(
        directory,
        cut=cut,
        search=search,
        max_labels=max_labels,
        optimal_order=optimal_order,
    )

    branch_stability = _make_branch_stability(data)
    rep_basket_scatter = _make_repertoire_basket_scatter(data)
    sku_behavior = _make_sku_behavior_map(data)
    cluster_evolution = _make_cluster_evolution(data)
    migration_matrix = _make_migration_matrix(data)
    migration_rank = _make_migration_rank(data)
    migration_ci = _make_migration_ci(data)
    retention_exit = _make_retention_exit_decomposition(data)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    coverage = float(data["report"]["customer_coverage"])
    quality = data["report"].get("cophenetic_correlation")
    quality_text = "N/A" if quality is None else f"{quality:.2f}"

    # Extended KPI cards
    cards = [
        ("CATEGORY CUSTOMERS", f"{data['report']['category_customers']:,}"),
        ("SKUS SHOWN", f"{len(data['ids']):,}"),
        ("CUSTOMER COVERAGE", f"{coverage:.1%}"),
        ("TREE FIT", quality_text),
        ("BOOTSTRAP REPLICATES", f"{data['report']['config']['bootstrap']}"),
        (
            "MEDIAN BRANCH SUPPORT",
            f"{float(np.nanmedian(data['nodes']['bootstrap_support'].to_numpy())):.0%}",
        ),
        ("RELIABLE MIGRATION SOURCES", f"{int(data['migration_reliable_a'].sum())}"),
        ("COPHENETIC CORRELATION", quality_text),
    ]

    card_html = "".join(
        (f'<div class="card"><small>{label}</small><strong>{value}</strong></div>')
        for label, value in cards
    )

    # Convert all figures to HTML
    figures = [
        ("Customer choice tree", tree, True),
        ("Customer repertoire similarity", heatmap, False),
        ("Branch stability", branch_stability, False),
        ("Repertoire vs basket affinity", rep_basket_scatter, False),
        ("SKU behavior map", sku_behavior, False),
        ("Migration matrix", migration_matrix, False),
        ("Top migration flows", migration_rank, False),
        ("Migration confidence intervals", migration_ci, False),
        ("Retention / exit decomposition", retention_exit, False),
    ]

    if cluster_evolution is not None:
        figures.append(("Cluster evolution", cluster_evolution, False))

    html_parts = []
    for _i, (_title, fig, include_js) in enumerate(figures):
        html_parts.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs=include_js,
                config={"responsive": True, "displaylogo": False},
            )
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
.section-title {{
    padding: 16px 20px;
    border-bottom: 1px solid #E5EAF2;
    font-size: 18px;
    font-weight: 600;
    color: {INK};
}}
.plotly-graph-div {{ width: 100%; }}
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
"""

    for i, (title, _, _) in enumerate(figures):
        page += f'<section><div class="section-title">{title}</div>{html_parts[i]}</section>\n'

    page += """<p class="note">
Shared customers suggest a common repertoire, not proven substitution.
Same-order association is a separate basket signal. Branch support comes
from whole-customer resampling. Migration rates describe observed
next-category-occasion transitions, not causal substitution.
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
