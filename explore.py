import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def _save_fig(fig, save_dir, filename):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in filename)
    path = os.path.join(save_dir, safe + '.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved: {path}")


def _to_serializable(v):
    """Converts value to a JSON-serializable type."""
    if v is None or (isinstance(v, float) and v != v):  # None or NaN
        return None
    if isinstance(v, (bool,)):
        return bool(v)
    if isinstance(v, (int,)):
        return int(v)
    if isinstance(v, (float,)):
        return float(v)
    if isinstance(v, str):
        return v
    # numpy scalar or other type
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
    except ImportError:
        pass
    return str(v)


def _iqr_outliers(series: pd.Series) -> dict:
    """Detects outliers using the IQR method for a numeric column."""
    s = series.dropna()
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    count = int(mask.sum())
    return {
        "method":      "IQR",
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "count":       count,
        "pct":         round(count / len(s) * 100, 2) if len(s) else 0.0,
    }


def column_report(df: pd.DataFrame, n_samples: int = 3) -> dict:
    """
    Returns a JSON-serializable dict with statistics and examples for each column.

    Parameters:
        df         : pandas DataFrame
        n_samples  : number of non-null sample values to return

    Output format (per column):
        {
          "dtype":      str,
          "non_null":   int,
          "null_count": int,
          "null_pct":   float,
          "n_unique":   int,
          "samples":    [val, ...],
          "outliers":   {method, lower_bound, upper_bound, count, pct} | null
        }
    outliers is null for non-numeric columns.
    """
    report = {}
    for col in df.columns:
        series = df[col]
        non_null_vals = series.dropna()
        k = min(n_samples, len(non_null_vals))
        samples = [_to_serializable(v) for v in non_null_vals.sample(k).tolist()]

        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        outliers = _iqr_outliers(series) if is_numeric else None

        report[col] = {
            "dtype":      str(series.dtype),
            "non_null":   int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "null_pct":   round(float(series.isna().mean() * 100), 2),
            "n_unique":   int(series.nunique()),
            "samples":    samples,
            "outliers":   outliers,
        }
    return report


def save_report(report: dict, path: str) -> None:
    """Saves the column_report to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {path}")


def report_summary(report: dict) -> pd.DataFrame:
    """
    Flattens the column_report() result into a concise DataFrame.

    Columns: dtype, non_null, null_count, null_pct, n_unique,
             outlier_count, outlier_pct (None for non-numeric columns).
    """
    rows = []
    for col, info in report.items():
        outliers = info.get("outliers") or {}
        rows.append({
            "column":       col,
            "dtype":        info["dtype"],
            "non_null":     info["non_null"],
            "null_count":   info["null_count"],
            "null_pct":     info["null_pct"],
            "n_unique":     info["n_unique"],
            "outlier_count": outliers.get("count"),
            "outlier_pct":   outliers.get("pct"),
        })
    return pd.DataFrame(rows).set_index("column")


def report_type_overview(report: dict) -> pd.DataFrame:
    """
    Compact overview of the column_report() result — one row per data type.

    Columns:
        n_cols        — number of columns of that type
        null_cols     — how many columns have at least one missing value
        avg_null_pct  — average % of missing values among columns of that type
        outlier_cols  — how many columns have outliers (numeric types only; NaN = n/a)
        columns       — list of column names
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for col, info in report.items():
        groups[info["dtype"]].append((col, info))

    rows = []
    for dtype, items in sorted(groups.items()):
        null_cols = sum(1 for _, info in items if info["null_count"] > 0)
        avg_null = round(sum(info["null_pct"] for _, info in items) / len(items), 2)
        outlier_vals = [
            info["outliers"]["count"]
            for _, info in items
            if info.get("outliers") is not None
        ]
        outlier_cols = sum(1 for v in outlier_vals if v > 0) if outlier_vals else None
        rows.append({
            "dtype":        dtype,
            "n_cols":       len(items),
            "null_cols":    null_cols,
            "avg_null_pct": avg_null,
            "outlier_cols": outlier_cols,
            "columns":      [col for col, _ in items],
        })
    return pd.DataFrame(rows).set_index("dtype")


def plot_null_bar(df: pd.DataFrame, min_pct: float = 0.0, figsize=(8, 6), save_dir=None) -> None:
    """
    Horizontal bar chart of the percentage of missing values (NaN) for each column.
    Columns with 0% NaN are skipped.

    Parameters:
        df      : pandas DataFrame
        min_pct : show only columns with null_pct > min_pct (default 0 = all with missing values)
        figsize : plot size
    """
    null_pct = (df.isna().mean() * 100).sort_values(ascending=True)
    null_pct = null_pct[null_pct > min_pct]

    if null_pct.empty:
        print("No columns with NaN values.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#d62728" if v > 50 else "#1f77b4" for v in null_pct.values]
    ax.barh(null_pct.index, null_pct.values, color=colors)
    ax.axvline(x=50, color="red", linestyle="--", linewidth=1, label="50% threshold")
    ax.set_xlabel("% missing (NaN)")
    ax.set_ylabel("Column")
    ax.set_title(f"Missing data per column ({len(null_pct)}/{df.shape[1]} columns with missing values)")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, save_dir, 'null_bar')
    plt.show()




def plot_outlier_bar(df: pd.DataFrame, min_pct: float = 0.0, figsize=(8, 6), save_dir=None) -> None:
    """
    Horizontal bar chart of the percentage of outliers (IQR method) for numeric columns.
    Columns with 0% outliers are skipped.

    Parameters:
        df      : pandas DataFrame
        min_pct : show only columns with outlier_pct > min_pct
        figsize : plot size
    """
    result = {}
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        info = _iqr_outliers(series)
        if info["pct"] > min_pct:
            result[col] = info["pct"]

    if not result:
        print("No columns with outliers meeting the min_pct criterion.")
        return

    outlier_pct = pd.Series(result).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(outlier_pct.index, outlier_pct.values, color="#ff7f0e")
    ax.set_xlabel("% outliers (IQR)")
    ax.set_ylabel("Column")
    ax.set_title(f"Outliers in numeric columns ({len(outlier_pct)} columns)")
    plt.tight_layout()
    _save_fig(fig, save_dir, 'outlier_bar')
    plt.show()



def _subtitle(series_full: pd.Series) -> str:
    n_total = len(series_full)
    n_null = int(series_full.isna().sum())
    return f"[{series_full.dtype}]  n={n_total:,}  NaN={n_null} ({n_null/n_total*100:.1f}%)"


def _unit_factor(series: pd.Series):
    """Returns (suffix, factor) based on the max absolute value of the series."""
    max_val = float(series.abs().max())
    if max_val >= 1e9:
        return " [B]", 1e9
    if max_val >= 1e6:
        return " [M]", 1e6
    if max_val >= 1e3:
        return " [K]", 1e3
    return "", 1.0


def plot_all_numeric(df: pd.DataFrame, cols=None, save_dir=None) -> pd.DataFrame:
    """
    Exploratory plots for numeric columns (excluding bool).
    Each column: histogram + boxplot.

    Histogram shows data WITHOUT outliers (IQR), if any exist.
    Boxplot shows ALL data — outliers visible as dots.
    X-axis auto-scaled (K / M / B) based on the clipped p1-p90 range.

    Returns: pd.DataFrame with statistics for all columns (min, max, mean, std, skewness).
    """
    if cols is None:
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]

    stats_rows = []

    for col in cols:
        if col not in df.columns:
            print(f"Column '{col}' does not exist — skipping.")
            continue

        series_full = df[col]
        series = series_full.dropna()

        # 1. Clip to p1-p90 on raw data
        p1, p90 = float(series.quantile(0.01)), float(series.quantile(0.80))
        if p1 < p90:
            hist_raw = series[(series >= p1) & (series <= p90)]
            n_clipped = len(series) - len(hist_raw)
            hist_title = (f"Histogram — middle 80%  "
                          f"(omitted {n_clipped:,} = "
                          f"{n_clipped/len(series)*100:.1f}% extreme values)")
        else:
            hist_raw = series
            hist_title = "Histogram"

        # 2. Determine scale from clipped range (unaffected by outliers)
        unit, factor = _unit_factor(hist_raw)
        xlabel = f"{col}{unit}"
        hist_series  = hist_raw / factor
        series_scaled = series  / factor

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"{col}  {_subtitle(series_full)}", fontsize=10)

        ax1.hist(hist_series, bins=50, color="#1f77b4", edgecolor="white", linewidth=0.4)
        ax1.axvline(hist_series.mean(),   color="red",   linestyle="--", linewidth=1,
                    label=f"mean={hist_series.mean():.3g}{unit}")
        ax1.axvline(hist_series.median(), color="green", linestyle="--", linewidth=1,
                    label=f"median={hist_series.median():.3g}{unit}")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Count")
        ax1.set_title(hist_title, fontsize=9)
        ax1.legend(fontsize=8)

        ax2.boxplot(series_scaled, vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#aec7e8"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
        ax2.set_xlabel(xlabel)
        ax2.set_title("Boxplot (all values)")
        ax2.set_yticks([])

        plt.tight_layout()
        _save_fig(fig, save_dir, f'numeric_{col}')
        plt.show()

        stats_rows.append({
            "column":   col,
            "min":      round(float(series.min()),  4),
            "max":      round(float(series.max()),  4),
            "mean":     round(float(series.mean()), 4),
            "median":   round(float(series.median()), 4),
            "std":      round(float(series.std()),  4),
            "skewness": round(float(series.skew()), 2),
            "n_null":   int(series_full.isna().sum()),
        })

    stats_df = pd.DataFrame(stats_rows).set_index("column")
    return stats_df


def plot_log_distribution(df: pd.DataFrame, cols=None, save_dir=None) -> pd.DataFrame:
    """
    Log-scale distribution plots for numeric columns (excluding bool).
    Each column: ECDF (X-axis = log1p) + Boxenplot (on log1p).

    Uses log(1+x) transformation — works for zero and small values too.

    Returns: pd.DataFrame with statistics (min, max, mean, median, std, skewness).
    """
    import numpy as np
    import seaborn as sns

    if cols is None:
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]

    stats_rows = []

    for col in cols:
        if col not in df.columns:
            print(f"Column '{col}' does not exist — skipping.")
            continue

        series_full = df[col]
        series = series_full.dropna()

        if len(series) == 0:
            print(f"Column '{col}' is empty — skipping.")
            continue

        # log1p — safe for non-negative values; clip negatives to 0
        s_log = np.log1p(series.clip(lower=0))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"{col}  {_subtitle(series_full)}", fontsize=10)

        def _ln_label(t):
            return f"ln(1+{np.expm1(t):.3g})"

        def _apply_log_ticks(ax, axis_dir='x'):
            """Sets ln(1+x) labels on bottom and original values on top.
            Filters ticks to the current axis range to avoid layout issues."""
            lim = ax.get_xlim() if axis_dir == 'x' else ax.get_ylim()
            raw = ax.get_xticks() if axis_dir == 'x' else ax.get_yticks()
            ticks = [t for t in raw if lim[0] <= t <= lim[1]]
            if not ticks:
                return
            ax.set_xticks(ticks)
            ax.set_xticklabels([_ln_label(t) for t in ticks], rotation=30, ha='right', fontsize=8)
            ax.set_xlim(lim)  # restore range after set_xticks

            top = ax.twiny()
            top.set_xlim(lim)
            top.set_xticks(ticks)
            top.set_xticklabels(
                [f"{np.expm1(t):.3g}" for t in ticks], fontsize=7, rotation=30, ha='left'
            )
            top.set_xlabel("original value", fontsize=8)

        # --- ECDF ---
        sorted_vals = np.sort(s_log)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax1.plot(sorted_vals, ecdf_y, color="#1f77b4", lw=1.5)
        ax1.axvline(float(s_log.mean()),   color="red",   linestyle="--", lw=1,
                    label=f"mean={float(series.mean()):.3g}")
        ax1.axvline(float(s_log.median()), color="green", linestyle="--", lw=1,
                    label=f"median={float(series.median()):.3g}")
        _apply_log_ticks(ax1)
        ax1.set_ylabel("CDF")
        ax1.set_title("ECDF  [log1p scale]")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # --- Boxenplot ---
        s_log_named = pd.Series(s_log.values, name=col)
        sns.boxenplot(x=s_log_named, ax=ax2, color="#aec7e8", linewidth=0.8)
        _apply_log_ticks(ax2)
        ax2.set_ylabel(col)
        ax2.set_title("Boxenplot  [log1p scale]")
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        _save_fig(fig, save_dir, f'log_dist_{col}')
        plt.show()

        stats_rows.append({
            "column":   col,
            "min":      round(float(series.min()),    4),
            "max":      round(float(series.max()),    4),
            "mean":     round(float(series.mean()),   4),
            "median":   round(float(series.median()), 4),
            "std":      round(float(series.std()),    4),
            "skewness": round(float(series.skew()),   2),
            "n_null":   int(series_full.isna().sum()),
        })

    return pd.DataFrame(stats_rows).set_index("column")


def plot_all_categorical(df: pd.DataFrame, cols=None,
                         chart: str = "bar", top_n: int = 15,
                         max_pie_unique: int = None, save_dir=None) -> None:
    """
    Exploratory plots for categorical and bool columns.

    Parameters:
        cols          : list of columns (None = all non-numeric + bool)
        chart         : "bar" (horizontal bars top-N) or "pie" (pie chart)
        top_n         : for "bar" — number of most frequent values to show
        max_pie_unique: for "pie" — max number of unique values for which
                        the chart is drawn; if None, uses top_n.
                        Pie always shows ALL values (sum = 100%).
    """
    if max_pie_unique is None:
        max_pie_unique = top_n

    if cols is None:
        cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])]

    for col in cols:
        if col not in df.columns:
            print(f"Column '{col}' does not exist — skipping.")
            continue

        series_full = df[col]
        series = series_full.dropna()
        all_counts = series.value_counts()
        n_unique = series.nunique()

        if chart == "pie":
            top_counts = all_counts.head(max_pie_unique)
            n_other_pie = int(all_counts.iloc[max_pie_unique:].sum()) if n_unique > max_pie_unique else 0
            if n_other_pie > 0:
                other_pct_pie = n_other_pie / len(series) * 100
                other_pie_label = f"other ({n_unique - max_pie_unique:,} values,  {other_pct_pie:.1f}%)"
                pie_values = list(top_counts.values) + [n_other_pie]
                pie_labels = list(top_counts.index.astype(str)) + [other_pie_label]
                n_slices = len(pie_values)
                colors = list(plt.cm.tab20.colors[:n_slices - 1]) + ["#cccccc"]
            else:
                pie_values = list(top_counts.values)
                pie_labels = list(top_counts.index.astype(str))
                colors = list(plt.cm.tab20.colors[:len(pie_values)])
            title = (f"{col}  {_subtitle(series_full)}  |  unique={n_unique:,}")
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(pie_values, labels=pie_labels,
                   autopct="%1.1f%%", startangle=140, colors=colors)
            ax.set_title(title, fontsize=9)
        else:
            counts = all_counts.head(top_n)
            n_other = int(all_counts.iloc[top_n:].sum()) if len(all_counts) > top_n else 0
            other_pct = n_other / len(series) * 100 if len(series) > 0 else 0.0
            other_label = (f"remaining {len(all_counts) - top_n:,} values = "
                           f"{n_other:,} occurrences ({other_pct:.1f}%)")
            title = (f"{col}  {_subtitle(series_full)}  |  "
                     f"unique={n_unique:,}  (top {len(counts)})")
            height = max(3, len(counts) * 0.45)
            fig, ax = plt.subplots(figsize=(9, height))
            ax.barh(counts.index[::-1].astype(str), counts.values[::-1],
                    color="#1f77b4")
            ax.set_xlabel("Count")
            ax.set_title(title, fontsize=9)
            if n_other > 0:
                ax.text(0.99, 0.01, other_label,
                        ha='right', va='bottom', fontsize=8.5,
                        color='#555555', style='italic',
                        transform=ax.transAxes)

        plt.tight_layout()
        _save_fig(fig, save_dir, f'categorical_{col}')
        plt.show()


def plot_post_transform(df: pd.DataFrame, top_n: int = 20, save_dir: str = None) -> None:
    """
    Visualizes columns after transformation, grouped by prefix.

    Binary groups (0/1) — bar chart of number of games with value 1, sorted descending:
      - dont_have_*          -> how many games lack the feature
      - supported_language_* -> language popularity
      - categories_*         -> category popularity
      - genres_*             -> genre popularity
      - tags_*               -> top_n most popular tags

    Columns *_is_top  -> 0 vs 1 for each: Developers, Publishers
    Columns *_frequency -> histogram of frequency distribution
    """
    n_total = len(df)

    def _is_binary(col):
        return set(df[col].dropna().unique()).issubset({0, 1})

    def _plot_binary_group(cols, title, filename, show_top_n=None):
        counts = df[cols].sum().sort_values(ascending=False)
        if show_top_n:
            counts = counts.head(show_top_n)
        fig, ax = plt.subplots(figsize=(11, max(3, len(counts) * 0.38)))
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color="#1f77b4")
        for bar, val in zip(bars, counts.values[::-1]):
            ax.text(bar.get_width() + n_total * 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}  ({val / n_total * 100:.1f}%)", va='center', fontsize=7.5)
        ax.set_xlabel("Number of games (value = 1)")
        ax.set_xlim(0, counts.max() * 1.22)
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        _save_fig(fig, save_dir, filename)
        plt.show()
        plt.close(fig)

    # --- Binary groups (dont_have_X appended to its group) ---
    binary_groups = [
        ('supported_language_', 'dont_have_supported_language', f"Supported languages — all languages",  'post_languages',  None),
        ('categories_',         'dont_have_categories',         f"Categories — top {top_n} + missing",   'post_categories', top_n),
        ('genres_',             'dont_have_genres',             f"Genres — all genres + missing",         'post_genres',     None),
        ('tags_',               'dont_have_tags',               f"Tags — top {top_n} + missing",          'post_tags',       top_n),
    ]

    for prefix, dont_have_col, title, filename, limit in binary_groups:
        cols = [c for c in df.columns if c.startswith(prefix) and _is_binary(c)]
        # dont_have appended at the end (will appear at top of chart after reversal)
        if dont_have_col in df.columns and _is_binary(dont_have_col):
            cols = cols + [dont_have_col]
        if cols:
            _plot_binary_group(cols, title, filename, show_top_n=limit)

    # --- is_top + dont_have: Developers and Publishers ---
    is_top_cols = [c for c in df.columns if c.endswith('_is_top')]
    if is_top_cols:
        # for each is_top find the corresponding dont_have and plot together
        for col in is_top_cols:
            prefix = col.replace('_is_top', '')
            dont_col = f'dont_have_{prefix}'
            sub_cols = [c for c in [dont_col, col] if c in df.columns and _is_binary(c)]
            if not sub_cols:
                continue
            fig, axes = plt.subplots(1, len(sub_cols), figsize=(5 * len(sub_cols), 4))
            if len(sub_cols) == 1:
                axes = [axes]
            labels_map = {col: ["Not top", "Top"], dont_col: ["Has data", "No data"]}
            for ax, c in zip(axes, sub_cols):
                counts = df[c].value_counts().reindex([0, 1], fill_value=0)
                bars = ax.bar(labels_map.get(c, ["0", "1"]), counts.values,
                              color=["#aec7e8", "#1f77b4"])
                for bar, val in zip(bars, counts.values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + n_total * 0.005,
                            f"{val:,}\n({val / n_total * 100:.1f}%)", ha='center', fontsize=9)
                ax.set_title(c)
                ax.set_ylabel("Number of games")
                ax.set_ylim(0, counts.max() * 1.2)
                ax.grid(axis='y', alpha=0.3)
            plt.suptitle(f"{prefix} — is_top and missing data", fontsize=11)
            plt.tight_layout()
            _save_fig(fig, save_dir, f'post_{prefix}_summary')
            plt.show()
            plt.close(fig)

    # --- frequency: distribution of cumulative frequency (boxenplot on log1p) ---
    freq_cols = [c for c in df.columns if c.endswith('_frequency')]
    if freq_cols:
        import numpy as np
        import seaborn as sns

        def _ln_label_freq(t):
            return f"ln(1+{np.expm1(t):.3g})"

        def _apply_log_ticks_freq(ax):
            lim = ax.get_xlim()
            raw = ax.get_xticks()
            ticks = [t for t in raw if lim[0] <= t <= lim[1]]
            if not ticks:
                return
            ax.set_xticks(ticks)
            ax.set_xticklabels([_ln_label_freq(t) for t in ticks], rotation=30, ha='right', fontsize=8)
            ax.set_xlim(lim)
            top = ax.twiny()
            top.set_xlim(lim)
            top.set_xticks(ticks)
            top.set_xticklabels([f"{np.expm1(t):.3g}" for t in ticks], fontsize=7, rotation=30, ha='left')
            top.set_xlabel("original value", fontsize=8)

        for col in freq_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            s_log = np.log1p(series.clip(lower=0))
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxenplot(x=s_log, ax=ax, color="#aec7e8", linewidth=0.8)
            _apply_log_ticks_freq(ax)
            ax.set_title(f"{col} — cumulative frequency distribution [log1p scale]")
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            _save_fig(fig, save_dir, f'post_{col}')
            plt.show()
            plt.close(fig)


def column_types_summary(df: pd.DataFrame):
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    print(f"Numeric columns ({len(num_cols)}):")
    print(list(num_cols))

    print(f"\nCategorical columns ({len(cat_cols)}):")
    print(list(cat_cols))

    return num_cols, cat_cols


def add_engineered_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Adds derived columns to the DataFrame:

    - n_languages     : sum of supported_language_* columns (excluding dont_have_supported_language)
    - n_categories    : sum of categories_* columns (excluding dont_have_categories)
    - n_genres        : sum of genres_* columns (excluding dont_have_genres)
    - n_tags          : sum of tags_* columns (excluding dont_have_tags)
    - engagement_ratio: Median playtime forever / (Average playtime forever + 1)
                        ratio near 1 -> engagement clustered around a typical player,
                        near 0 -> heavy-tailed distribution (hardcore minority)

    Parameters:
        df      : pandas DataFrame (result of the transformation pipeline)
        inplace : True = modifies df in place, False (default) = returns a copy
    """
    result = df if inplace else df.copy()

    lang_cols = [c for c in result.columns if c.startswith("supported_language_")]
    cat_cols  = [c for c in result.columns if c.startswith("categories_")]
    gen_cols  = [c for c in result.columns if c.startswith("genres_")]
    tag_cols  = [c for c in result.columns if c.startswith("tags_")]

    result["n_languages"]  = result[lang_cols].sum(axis=1)
    result["n_categories"] = result[cat_cols].sum(axis=1)
    result["n_genres"]     = result[gen_cols].sum(axis=1)
    result["n_tags"]       = result[tag_cols].sum(axis=1)

    result["engagement_ratio"] = (
        result["Median playtime forever"] / (result["Average playtime forever"] + 1)
    )

    print(f"Added columns: n_languages ({len(lang_cols)} flags), "
          f"n_categories ({len(cat_cols)} flags), "
          f"n_genres ({len(gen_cols)} flags), "
          f"n_tags ({len(tag_cols)} flags), engagement_ratio")

    return result
