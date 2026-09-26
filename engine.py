"""Transaction-only SKU repertoire, basket and sequential category-occasion engine.

python engine.py --input sample_data/instacart_yogurt.parquet --out output/yogurt
CSV IDs are read as text; Parquet identifier types are preserved. No causal substitution.
The previous visual.py/app.py require updating for schema_version 3 before use.
"""

from __future__ import annotations

import argparse
import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform
from scipy.stats import hypergeom


@dataclass(frozen=True)
class Config:
    input_path: str
    out: str = "output/choice_run"
    user_col: str = "user_id"
    order_col: str = "order_id"
    product_col: str = "product_id"
    name_col: str = "product_name"
    sequence_col: str = "order_number"
    category_col: str | None = "aisle"
    category_value: str | None = "yogurt"
    date_col: str | None = None
    analysis_start: str | None = None
    analysis_end: str | None = None
    min_buyers: int = 30
    max_skus: int = 120
    max_input_rows: int = 5_000_000
    bootstrap: int = 30
    seed: int = 42
    linkage_method: str = "average"
    switching: bool = True
    strict_names: bool = False
    min_exit_opportunities: int = 50
    stability_cut: float = 0.7
    cluster_counts: tuple[int, ...] = (4, 6, 8)


def prepare_data(c: Config):
    path = Path(c.input_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(path)
    elif path.suffix.lower() == ".csv":
        lf = pl.scan_csv(path, infer_schema=False)
    else:
        raise ValueError("CSV or Parquet required")
    schema = lf.collect_schema()
    required = {"user_id": c.user_col, "order_id": c.order_col, "product_id": c.product_col}
    if c.switching:
        required["order_number"] = c.sequence_col
    if any(v not in schema for v in required.values()):
        raise ValueError("Missing required mapped column")
    if len(set(required.values())) != len(required):
        raise ValueError("Mapped required columns must be different")
    if c.category_value is not None:
        if c.category_col not in schema:
            raise ValueError("Missing category column")
        lf = lf.filter(pl.col(c.category_col).cast(pl.String) == c.category_value)
    if c.analysis_start or c.analysis_end:
        if c.date_col not in schema:
            raise ValueError("Date window requires date_col")

        dates = [
            datetime.date.fromisoformat(v) if v else None
            for v in (c.analysis_start, c.analysis_end)
        ]
        if dates[0] and dates[1] and dates[0] > dates[1]:
            raise ValueError("Reversed date window")
        d = pl.col(c.date_col).cast(pl.String).str.to_date(strict=True)
        if dates[0]:
            lf = lf.filter(d >= pl.lit(dates[0]))
        if dates[1]:
            lf = lf.filter(d <= pl.lit(dates[1]))
    fields = [
        pl.col(v).str.strip_chars().alias(k) if schema[v] == pl.String else pl.col(v).alias(k)
        for k, v in required.items()
        if k != "order_number"
    ]
    if c.switching:
        fields.append(pl.col(c.sequence_col).cast(pl.Int64, strict=True).alias("order_number"))
    if c.name_col in schema:
        fields.append(pl.col(c.name_col).cast(pl.String).alias("product_name"))
    df = lf.select(fields).limit(c.max_input_rows + 1).collect()
    if df.height > c.max_input_rows or df.is_empty():
        raise ValueError("Empty or over configured row limit")
    if df.select(pl.any_horizontal([pl.col(k).is_null() for k in required]).any()).item():
        raise ValueError("Required fields cannot be null")
    for k in ("user_id", "order_id", "product_id"):
        if df[k].dtype == pl.String and df.filter(pl.col(k) == "").height:
            raise ValueError(f"Blank {k}")
    if c.switching and df.filter(pl.col("order_number") <= 0).height:
        raise ValueError("order_number must be positive")
    if (
        df.group_by("order_id")
        .agg(pl.col("user_id").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    ):
        raise ValueError("Order ID belongs to multiple users")
    if c.switching:
        for keys, value in [
            ("order_id", "order_number"),
            (["user_id", "order_number"], "order_id"),
        ]:
            if (
                df.group_by(keys)
                .agg(pl.col(value).n_unique().alias("n"))
                .filter(pl.col("n") > 1)
                .height
            ):
                raise ValueError("Inconsistent order sequence")
    conflicts = 0
    if "product_name" in df.columns:
        conflicts = (
            df.group_by("product_id")
            .agg(pl.col("product_name").drop_nulls().n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        )
        if c.strict_names and conflicts:
            raise ValueError(f"{conflicts} conflicting SKU names")
    before = df.height
    df = df.unique(subset=["order_id", "product_id"], keep="first")
    return df, {
        "rows_before_dedup": before,
        "rows_after_dedup": df.height,
        "name_conflict_skus": conflicts,
        "id_types": {k: str(df[k].dtype) for k in ("user_id", "order_id", "product_id")},
    }


def _matrix(df, skus, key):
    ids, rows = np.unique(df[key].to_numpy(), return_inverse=True)
    lookup = {v: i for i, v in enumerate(skus)}
    cols = np.fromiter((lookup[v] for v in df["product_id"]), dtype=np.int32, count=df.height)
    m = coo_matrix(
        (np.ones(df.height, dtype=np.int64), (rows, cols)), shape=(len(ids), len(skus))
    ).tocsr()
    m.data[:] = 1
    return m, ids


def _overlap(m, weights=None):
    return (m.T @ (m if weights is None else m.multiply(weights[:, None]))).toarray()


def _jac(counts):
    diag = np.diag(counts)
    den = diag[:, None] + diag[None, :] - counts
    j = np.divide(counts, den, out=np.zeros_like(counts, dtype=float), where=den > 0)
    np.fill_diagonal(j, 1.0)
    return j


def _clades(z, p):
    nodes = {i: frozenset([i]) for i in range(p)}
    result = []
    for t, (a, b, _, _) in enumerate(z):
        v = nodes[int(a)] | nodes[int(b)]
        nodes[p + t] = v
        result.append(v)
    return result


def _stability(x, z, c):
    p = x.shape[1]
    branch = np.full(p - 1, np.nan)
    co = np.full((p, p), np.nan)
    if c.bootstrap == 0:
        return branch, co
    hits = np.zeros(p - 1, dtype=int)
    co = np.zeros((p, p), dtype=float)
    root = _clades(z, p)
    rng = np.random.default_rng(c.seed)

    # Pre-allocate Jaccard buffer for bootstrap loop
    jac_buf = np.empty((p, p), dtype=float)

    for _ in range(c.bootstrap):
        w = np.bincount(rng.integers(x.shape[0], size=x.shape[0]), minlength=x.shape[0]).astype(
            np.int64
        )
        np.subtract(1.0, _jac(_overlap(x, w)), out=jac_buf)
        zz = linkage(squareform(jac_buf, checks=False), method=c.linkage_method)
        present = set(_clades(zz, p))
        hits += np.fromiter((v in present for v in root), dtype=np.int32, count=p - 1)
        labels = fcluster(zz, t=c.stability_cut, criterion="distance")
        co += (labels[:, None] == labels[None, :]) / c.bootstrap
    branch[:-1] = hits[:-1] / c.bootstrap  # Root is structurally guaranteed.
    return branch, co


def build_transitions(full, skus):
    """Strict new-entry migration, retained-with-new-entry expansion, full-category denominators."""
    basket = (
        full.group_by(["user_id", "order_id", "order_number"])
        .agg(pl.col("product_id").unique().sort().alias("items"))
        .sort(["user_id", "order_number", "order_id"])
    )
    selected = set(skus)
    fields = [
        "opportunities",
        "retained",
        "exits",
        "exit_new_selected",
        "exit_new_unselected_only",
        "exit_no_new",
    ]
    state = {field: dict.fromkeys(skus, 0) for field in fields}
    migration = {}
    expansion = {}
    exit_events = []
    pair_events = []

    # Vectorized consecutive-order self-join using Polars list expressions
    basket_lf = basket.lazy()
    prev = basket_lf.rename(
        {
            "order_number": "prev_number",
            "items": "prev_items",
            "order_id": "prev_order_id",
        }
    ).select(["user_id", "prev_number", "prev_items"])

    pairs_lf = basket_lf.join(
        prev,
        left_on=["user_id", pl.col("order_number") - 1],
        right_on=["user_id", "prev_number"],
        how="inner",
    )

    # Compute set operations using Polars list expressions
    pairs_lf = pairs_lf.with_columns(
        [
            pl.col("items").list.set_difference(pl.col("prev_items")).alias("added"),
            pl.col("items").list.set_intersection(pl.col("prev_items")).alias("retained_items"),
        ]
    )

    # Filter to selected SKUs only
    selected_list = list(selected)
    pairs_lf = pairs_lf.with_columns(
        [
            pl.col("added").list.filter(pl.element().is_in(selected_list)).alias("new_selected"),
            pl.col("prev_items")
            .list.filter(pl.element().is_in(selected_list))
            .alias("prev_selected"),
        ]
    )

    pairs_df = pairs_lf.collect()
    occasion_pairs = pairs_df.height

    if occasion_pairs > 0:
        # Compute retained = prev_selected ∩ retained_items
        pairs_df = pairs_df.with_columns(
            retained=pl.col("prev_selected").list.set_intersection(pl.col("retained_items"))
        )

        # Aggregate opportunities: count of prev_selected per SKU
        opps = pairs_df.explode("prev_selected").group_by("prev_selected").len()
        for row in opps.iter_rows():
            state["opportunities"][row[0]] += row[1]

        # Aggregate retained: count of retained per SKU
        ret = pairs_df.explode("retained").group_by("retained").len()
        for row in ret.iter_rows():
            state["retained"][row[0]] += row[1]

        # For each pair, compute exits = prev_selected - retained
        pairs_df = pairs_df.with_columns(
            exited=pl.col("prev_selected").list.set_difference(pl.col("retained"))
        )

        # Aggregate exits
        ex = pairs_df.explode("exited").group_by("exited").len()
        for row in ex.iter_rows():
            state["exits"][row[0]] += row[1]

        # For each pair, determine kind based on new_selected and added
        pairs_df = pairs_df.with_columns(
            kind=pl.when(pl.col("new_selected").list.len() > 0)
            .then(pl.lit("exit_new_selected"))
            .when(pl.col("added").list.len() > 0)
            .then(pl.lit("exit_new_unselected_only"))
            .otherwise(pl.lit("exit_no_new"))
        )

        # Process exits: for each exited SKU, apply the pair's kind
        exited_df = pairs_df.explode("exited").select(["user_id", "exited", "kind", "new_selected"])
        if exited_df.height > 0:
            ex_kind = exited_df.group_by(["exited", "kind"]).len()
            for a, kind, count in ex_kind.iter_rows():
                state[kind][a] += count

            # Build exit_events and migration/pair_events
            for row in exited_df.iter_rows(named=True):
                user, a, kind = row["user_id"], row["exited"], row["kind"]
                exit_events.append((user, a))
                for b in row["new_selected"]:
                    migration[a, b] = migration.get((a, b), 0) + 1
                    pair_events.append((user, a, b))

        # Expansion: cross product of retained a and new_selected b
        for row in pairs_df.iter_rows(named=True):
            retained = row["retained"]
            new_selected = row["new_selected"]
            if retained and new_selected:
                for a in retained:
                    for b in new_selected:
                        expansion[a, b] = expansion.get((a, b), 0) + 1

    return dict(
        state,
        migration=migration,
        expansion=expansion,
        exit_events=exit_events,
        pair_events=pair_events,
        occasion_pairs=occasion_pairs,
    )


def _bh(p):
    idx = np.argsort(p)
    q = np.empty(len(p))
    q[idx] = np.minimum(
        1, np.minimum.accumulate((p[idx] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    )
    return q


def _transition_intervals(trans, skus, users, c):
    p = len(skus)
    lo = np.full((p, p), np.nan)
    hi = lo.copy()
    if c.bootstrap == 0 or not trans or not trans["exit_events"]:
        return lo, hi

    # Guard against excessive memory usage from large p*p migration matrix
    max_pair_events = 1_000_000
    if len(trans["pair_events"]) > max_pair_events:
        # Too many pair events for sparse p*p matrix; return NaN intervals
        return lo, hi

    u = {v: i for i, v in enumerate(users)}
    s = {v: i for i, v in enumerate(skus)}
    ee = trans["exit_events"]
    pe = trans["pair_events"]
    ex = coo_matrix(
        (np.ones(len(ee), dtype=np.int64), ([u[who] for who, a in ee], [s[a] for who, a in ee])),
        shape=(len(users), p),
    ).tocsr()
    mig = coo_matrix(
        (
            np.ones(len(pe), dtype=np.int64),
            ([u[who] for who, a, b in pe], [s[a] * p + s[b] for who, a, b in pe]),
        ),
        shape=(len(users), p * p),
    ).tocsr()
    rng = np.random.default_rng(c.seed + 1)
    values = np.full((c.bootstrap, p, p), np.nan)
    for t in range(c.bootstrap):
        w = np.bincount(rng.integers(len(users), size=len(users)), minlength=len(users))
        den = np.asarray(ex.T @ w).reshape(p)
        num = np.asarray(mig.T @ w).reshape(p, p)
        values[t] = np.divide(
            num, den[:, None], out=np.full((p, p), np.nan), where=den[:, None] > 0
        )
    for i, a in enumerate(skus):
        if trans["exits"][a] < c.min_exit_opportunities:
            continue
        for j in range(p):
            valid = values[:, i, j]
            valid = valid[np.isfinite(valid)]
            if len(valid):
                lo[i, j], hi[i, j] = np.percentile(valid, [2.5, 97.5])
    return lo, hi


def _clusters(z, j, skus, buyers, counts):
    p = len(skus)
    rows = []
    assignments = {}
    actual = {}
    for k in sorted(set(counts)):
        if not 2 <= k < p:
            continue
        labels = fcluster(z, t=k, criterion="maxclust")
        assignments[k] = labels
        actual[str(k)] = len(set(labels))
        for cl in sorted(set(labels)):
            members = np.flatnonzero(labels == cl)
            intra = j[np.ix_(members, members)]
            tri = intra[np.triu_indices(len(members), 1)]
            rows.append(
                {
                    "requested_k": k,
                    "actual_k": actual[str(k)],
                    "cluster_id": int(cl),
                    "sku_count": len(members),
                    "buyer_count_sum_nonunique": int(buyers[members].sum()),
                    "mean_within_repertoire_jaccard": float(tri.mean()) if len(tri) else None,
                    "member_product_ids": json.dumps([str(skus[i]) for i in members]),
                }
            )
    return rows, assignments, actual


def run(c: Config):
    if (
        c.min_buyers < 1
        or c.max_skus < 2
        or c.max_input_rows < 1
        or c.bootstrap < 0
        or c.min_exit_opportunities < 1
        or not 0 <= c.stability_cut <= 1
    ):
        raise ValueError("Invalid numerical configuration")
    if c.linkage_method not in ("average", "complete", "single"):
        raise ValueError("Unsupported linkage")
    started = perf_counter()
    full, quality = prepare_data(c)
    category_users = full["user_id"].n_unique()
    category_orders = full["order_id"].n_unique()
    chosen = (
        full.group_by("product_id")
        .agg(pl.col("user_id").n_unique().alias("buyers"))
        .filter(pl.col("buyers") >= c.min_buyers)
        .sort(["buyers", "product_id"], descending=[True, False])
        .head(c.max_skus)
    )
    if chosen.height < 2:
        raise ValueError("Fewer than two qualifying SKUs")
    skus = sorted(chosen["product_id"].to_list())
    p = len(skus)
    selected = full.filter(pl.col("product_id").is_in(skus))
    xu, kept_users = _matrix(selected, skus, "user_id")
    xo, kept_orders = _matrix(selected, skus, "order_id")
    uc = _overlap(xu)
    oc = _overlap(xo)
    j = _jac(uc)
    jo = _jac(oc)
    distance = 1 - j
    np.fill_diagonal(distance, 0)
    condensed = squareform(distance, checks=True)
    z = linkage(condensed, method=c.linkage_method)
    support, co = _stability(xu, z, c)
    trans = build_transitions(full, skus) if c.switching else None
    lo, hi = _transition_intervals(trans, skus, np.unique(full["user_id"].to_numpy()), c)
    names = {}
    if "product_name" in full.columns:
        freq = (
            full.filter(pl.col("product_name").is_not_null())
            .group_by(["product_id", "product_name"])
            .len()
            .sort(["product_id", "len", "product_name"], descending=[False, True, False])
            .unique(subset=["product_id"], keep="first")
        )
        names = dict(freq.select(["product_id", "product_name"]).iter_rows())
    buyers = np.diag(uc)
    orders = np.diag(oc)

    def _st(field: str, s: str) -> int:
        return trans[field][s] if trans else 0

    # SKU-level counts
    trans_opps = np.array([_st("opportunities", s) for s in skus])
    exit_opps = np.array([_st("exits", s) for s in skus])
    retained = np.array([_st("retained", s) for s in skus])
    exit_new_sel = np.array([_st("exit_new_selected", s) for s in skus])
    exit_new_unsel = np.array([_st("exit_new_unselected_only", s) for s in skus])
    exit_no_new = np.array([_st("exit_no_new", s) for s in skus])

    # SKU-level rates (safe division)
    def _safe_rate(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0)

    retention_rate = _safe_rate(retained, trans_opps)
    exit_rate = _safe_rate(exit_opps, trans_opps)
    exit_new_selected_rate = _safe_rate(exit_new_sel, exit_opps)
    exit_new_unselected_only_rate = _safe_rate(exit_new_unsel, exit_opps)
    exit_no_new_rate = _safe_rate(exit_no_new, exit_opps)

    sku = pl.DataFrame(
        {
            "product_id": skus,
            "product_name": [names.get(s) or str(s) for s in skus],
            "buyers": buyers,
            "category_orders": orders,
            "buyer_penetration": buyers / category_users,
            "order_support": orders / category_orders,
            "transition_opportunities": trans_opps,
            "exit_opportunities": exit_opps,
            "retention_occasions": retained,
            "exit_new_selected": exit_new_sel,
            "exit_new_unselected_only": exit_new_unsel,
            "exit_no_new_sku": exit_no_new,
            "retention_rate": retention_rate,
            "exit_rate": exit_rate,
            "exit_new_selected_rate": exit_new_selected_rate,
            "exit_new_unselected_only_rate": exit_new_unselected_only_rate,
            "exit_no_new_rate": exit_no_new_rate,
        }
    )
    a, b = np.triu_indices(p, 1)
    su = uc[a, b]
    so = oc[a, b]
    ea = np.array([_st("exits", skus[i]) for i in a])
    eb = np.array([_st("exits", skus[i]) for i in b])
    mig = trans["migration"] if trans else {}
    expand = trans["expansion"] if trans else {}

    def directed(table, aa, bb):
        return np.fromiter(
            (table.get((skus[i], skus[k]), 0) for i, k in zip(aa, bb, strict=True)),
            dtype=np.int64,
            count=len(a),
        )

    ab = directed(mig, a, b)
    ba = directed(mig, b, a)

    def rate(num, den):
        return np.divide(
            num, den, out=np.full(len(a), np.nan), where=den >= c.min_exit_opportunities
        )

    pvals = hypergeom.sf(su - 1, category_users, buyers[a], buyers[b])
    pair = pl.DataFrame(
        {
            "product_id_a": [skus[i] for i in a],
            "product_id_b": [skus[i] for i in b],
            "buyers_a": buyers[a],
            "buyers_b": buyers[b],
            "orders_a": orders[a],
            "orders_b": orders[b],
            "shared_customers": su,
            "shared_orders": so,
            "customer_jaccard": j[a, b],
            "repertoire_distance": distance[a, b],
            "bootstrap_pair_cocluster": co[a, b],
            "customer_overlap_lift": su * category_users / (buyers[a] * buyers[b]),
            "expected_shared_customers": buyers[a] * buyers[b] / category_users,
            "excess_shared_customers": su - buyers[a] * buyers[b] / category_users,
            "order_jaccard": jo[a, b],
            "basket_support": so / category_orders,
            "confidence_a_to_b": so / orders[a],
            "confidence_b_to_a": so / orders[b],
            "basket_lift": so * category_orders / (orders[a] * orders[b]),
            "basket_leverage": so / category_orders
            - (orders[a] / category_orders) * (orders[b] / category_orders),
            "overlap_p_exploratory": pvals,
            "overlap_q_exploratory": _bh(pvals),
            "migration_a_to_b": ab,
            "migration_b_to_a": ba,
            "migration_rate_a_to_b": rate(ab, ea),
            "migration_rate_b_to_a": rate(ba, eb),
            "migration_ci_lower_a_to_b": lo[a, b],
            "migration_ci_upper_a_to_b": hi[a, b],
            "migration_ci_lower_b_to_a": lo[b, a],
            "migration_ci_upper_b_to_a": hi[b, a],
            "migration_reliable_a": ea >= c.min_exit_opportunities,
            "migration_reliable_b": eb >= c.min_exit_opportunities,
            "expansion_a_to_b": directed(expand, a, b),
            "expansion_b_to_a": directed(expand, b, a),
            "expansion_rate_a_to_b": _safe_rate(
                np.array([expand.get((skus[i], skus[k]), 0) for i, k in zip(a, b, strict=True)]),
                retained[a],
            ),
            "expansion_rate_b_to_a": _safe_rate(
                np.array([expand.get((skus[k], skus[i]), 0) for i, k in zip(a, b, strict=True)]),
                retained[b],
            ),
        }
    )
    nodes = pl.DataFrame(
        {
            "node_id": np.arange(p, 2 * p - 1),
            "left": z[:, 0].astype(np.int64),
            "right": z[:, 1].astype(np.int64),
            "height": z[:, 2],
            "leaf_count": z[:, 3].astype(np.int64),
            "bootstrap_support": support,
        }
    )
    profiles, assignments, actual = _clusters(z, j, skus, buyers, c.cluster_counts)
    out = Path(c.out)
    out.mkdir(parents=True, exist_ok=True)
    sku.write_parquet(out / "sku_metrics.parquet")
    pair.write_parquet(out / "pair_metrics.parquet")
    nodes.write_parquet(out / "tree_nodes.parquet")
    if profiles:
        pl.DataFrame(profiles).write_parquet(out / "cluster_profiles.parquet")
    if assignments:
        pl.DataFrame(
            {"product_id": skus, **{f"cluster_k{k}": v for k, v in assignments.items()}}
        ).write_parquet(out / "cluster_assignments.parquet")
    fit = (
        float(cophenet(z, condensed)[0]) if np.std(condensed) > 0 and np.std(z[:, 2]) > 0 else None
    )
    if fit is not None and not np.isfinite(fit):
        fit = None
    report = {
        "schema_version": "3",
        "config": asdict(c),
        "versions": {"polars": pl.__version__},
        "data_quality": quality,
        "category_customers": category_users,
        "category_orders": category_orders,
        "retained_customers": len(kept_users),
        "retained_orders": len(kept_orders),
        "retained_skus": p,
        "customer_coverage": len(kept_users) / category_users,
        "order_coverage": len(kept_orders) / category_orders,
        "transition_occasion_pairs": trans["occasion_pairs"] if trans else 0,
        "sku_selection_method": "minimum buyers then top-N distinct buyers",
        "cluster_counts_requested_vs_actual": actual,
        "cophenetic_correlation": fit,
        "leaf_product_ids": [skus[i] for i in dendrogram(z, no_plot=True)["leaves"]],
        "runtime_seconds": round(perf_counter() - started, 3),
        "interpretation": "Repertoire similarity is not choice probability or causal substitution. Migration is strict new entry after A exit on next observed category occasion; rates can sum above 1.",
    }
    (out / "run.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, dest="input_path")
    ap.add_argument("--out", default="output/choice_run")
    ap.add_argument("--category-col", default="aisle")
    ap.add_argument("--category", dest="category_value", default="yogurt")
    ap.add_argument("--min-buyers", type=int, default=30)
    ap.add_argument("--max-skus", type=int, default=120)
    ap.add_argument("--bootstrap", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-exit-opportunities", type=int, default=50)
    ap.add_argument("--stability-cut", type=float, default=0.7)
    ap.add_argument("--date-col")
    ap.add_argument("--analysis-start")
    ap.add_argument("--analysis-end")
    ap.add_argument(
        "--linkage",
        dest="linkage_method",
        choices=("average", "complete", "single"),
        default="average",
    )
    ap.add_argument("--no-switching", action="store_true")
    args = vars(ap.parse_args())
    args["switching"] = not args.pop("no_switching")
    for key in ("category_col", "category_value"):
        if args[key] == "none":
            args[key] = None
    print(json.dumps(run(Config(**args)), indent=2, default=str))


if __name__ == "__main__":
    main()
