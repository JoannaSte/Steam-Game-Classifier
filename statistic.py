import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def _save_fig(fig, save_dir, filename):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in filename)
    path = os.path.join(save_dir, safe + '.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved: {path}")


def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def correlate_features(
    df: pd.DataFrame,
    targets: list,
    features: list,
    figsize: tuple = None,
    annot_threshold: float = 0.0,
    top_n: int = None,
    save_dir: str = None,
) -> pd.DataFrame:
    """
    Spearman rank correlation between a list of features and a list of targets.

    Returns a DataFrame of rho values and displays a heatmap.

    Parameters:
        df               : pandas DataFrame
        targets          : target columns, e.g. ["Positive", "Negative"]
        features         : feature columns to correlate
        figsize          : plot size (auto-sized if None)
        annot_threshold  : only annotate cells where |rho| >= threshold (0..1)
        top_n            : limit plot to top_n features by mean |rho|
        save_dir         : directory to save the plot (None = don't save)
    """
    missing = [c for c in targets + features if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    # allow user to pass e.g. 30 instead of 0.30
    if annot_threshold > 1.0:
        annot_threshold = annot_threshold / 100.0

    rho_data = {}
    pval_data = {}

    for target in targets:
        rhos, pvals = [], []
        for feat in features:
            data = df[[target, feat]].dropna()
            if len(data) < 30:
                rhos.append(np.nan)
                pvals.append(np.nan)
            else:
                rho, p = stats.spearmanr(data[target], data[feat])
                rhos.append(round(float(rho), 3))
                pvals.append(p)
        rho_data[target]  = rhos
        pval_data[target] = pvals

    rho_df  = pd.DataFrame(rho_data,  index=features)
    pval_df = pd.DataFrame(pval_data, index=features)

    plot_rho  = rho_df
    plot_pval = pval_df
    if top_n is not None:
        row_order = rho_df.abs().mean(axis=1).sort_values(ascending=False).index
        plot_rho  = rho_df.loc[row_order].head(top_n)
        plot_pval = pval_df.loc[row_order].head(top_n)

    n_feats, n_targets = plot_rho.shape
    if figsize is None:
        figsize = (max(4, n_targets * 2.2), max(5, n_feats * 0.45))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(plot_rho.abs().max().max(), 0.1)

    im = ax.imshow(plot_rho.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(n_targets))
    ax.set_xticklabels(plot_rho.columns, fontsize=10)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(plot_rho.index, fontsize=8)
    title = "Spearman Correlation: features vs. popularity targets"
    if top_n is not None:
        title += f" (top {n_feats})"
    ax.set_title(title, fontsize=11)

    for i in range(n_feats):
        for j in range(n_targets):
            rho = plot_rho.iloc[i, j]
            p   = plot_pval.iloc[i, j]
            if pd.isna(rho) or abs(rho) < annot_threshold:
                continue
            sig = _sig(p) if not pd.isna(p) else ""
            color = "white" if abs(rho) > vmax * 0.6 else "black"
            ax.text(j, i, f"{rho:.2f}{sig}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    _save_fig(fig, save_dir, 'correlate_features')
    plt.show()

    return rho_df


def mann_whitney_binary_scan(
    df: pd.DataFrame,
    targets: list,
    binary_cols: list = None,
    min_n_per_group: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Mann-Whitney U test for each binary column (0/1) × each target.

    Designed for one-hot encoded columns (genres_action, categories_co_op, Mac, Linux, etc.).
    Measures rank-biserial effect size r: +1 means col=1 group always higher.
    binary_cols=None → auto-detect all columns with exactly two values {0, 1}.

    Parameters:
        df              : pandas DataFrame
        targets         : list of numeric target columns
        binary_cols     : list of binary columns to test (None = auto-detect)
        min_n_per_group : minimum group size required to run the test
        verbose         : print summary table
    """
    if binary_cols is None:
        binary_cols = [
            c for c in df.columns
            if c not in targets
            and df[c].dropna().isin([0, 1]).all()
            and df[c].nunique() == 2
        ]

    rows = []
    for col in binary_cols:
        g0 = df[df[col] == 0]
        g1 = df[df[col] == 1]
        for target in targets:
            v0 = g0[target].dropna().values
            v1 = g1[target].dropna().values
            if len(v0) < min_n_per_group or len(v1) < min_n_per_group:
                continue
            stat, p = stats.mannwhitneyu(v0, v1, alternative="two-sided")
            r = 1 - (2 * float(stat)) / (len(v0) * len(v1))
            rows.append({
                "col":      col,
                "target":   target,
                "n_0":      len(v0),
                "n_1":      len(v1),
                "median_0": round(float(np.median(v0)), 2),
                "median_1": round(float(np.median(v1)), 2),
                "effect_r": round(r, 4),
                "U":        round(float(stat), 0),
                "p_value":  p,
                "sig":      _sig(p),
            })

    result_df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    if verbose:
        print(f"[mann_whitney_binary_scan]  binary cols={len(binary_cols)}  "
              f"targets={len(targets)}  tests={len(rows)}\n")
        print(result_df.head(20).to_string(index=False))
    return result_df


def plot_mann_whitney_binary_heatmap(
    mw_df: pd.DataFrame,
    value_col: str = "effect_r",
    top_n: int = 40,
    figsize: tuple = None,
    save_dir: str = None,
) -> None:
    """
    Heatmap of mann_whitney_binary_scan results.

    X-axis: targets, Y-axis: binary columns (top_n by mean |r|).
    Color: rank-biserial r (−1..+1). Annotations: r value + significance stars.

    Parameters:
        mw_df     : DataFrame from mann_whitney_binary_scan
        value_col : column to use for color (default 'effect_r')
        top_n     : number of rows to display, sorted by mean |r|
        figsize   : plot size (auto-sized if None)
        save_dir  : directory to save the plot
    """
    pivot_val = mw_df.pivot_table(index="col", columns="target",
                                  values=value_col, aggfunc="first")
    pivot_sig = mw_df.pivot_table(index="col", columns="target",
                                  values="sig", aggfunc="first")

    row_order = pivot_val.abs().mean(axis=1).sort_values(ascending=False).index
    pivot_val = pivot_val.loc[row_order].head(top_n)
    pivot_sig = pivot_sig.loc[row_order].head(top_n)

    n_rows, n_cols = pivot_val.shape
    if figsize is None:
        figsize = (max(5, n_cols * 2.5), max(4, n_rows * 0.45 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(pivot_val.abs().max().max(), 0.1)
    im = ax.imshow(pivot_val.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Rank-biserial r (effect size)", fontsize=9)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot_val.columns, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot_val.index, fontsize=9)
    ax.set_title(f"Mann-Whitney U: binary features vs. targets (top {top_n} by |r|)", fontsize=11)

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot_val.iloc[i, j]
            sig = pivot_sig.iloc[i, j]
            if pd.isna(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}\n{sig}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    _save_fig(fig, save_dir, 'mann_whitney_binary_heatmap')
    plt.show()
