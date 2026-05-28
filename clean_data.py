import numpy as np
import pandas as pd
from explore import _iqr_outliers

_EMPTY_VALUES = {"", "[]", "none", "null", "nan", "n/a", "na"}


def _is_empty(val) -> bool:
    """Returns True for NaN, None, empty lists/dicts/strings and common placeholders."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, (list, dict, set)) and len(val) == 0:
        return True
    if isinstance(val, str) and val.strip().lower() in _EMPTY_VALUES:
        return True
    return False


def drop_high_null_columns(df, threshold, verbose=True):
    """
    Drops columns where the percentage of empty values exceeds threshold.

    Considered empty: NaN, None, empty lists [], empty strings "",
    and common placeholders: "none", "null", "nan", "n/a", "na".

    Parameters:
        df         : pandas DataFrame
        threshold  : threshold in percent (e.g. 50 means above 50% empty)
        verbose    : whether to print which columns were dropped
    """
    empty_pct = df.apply(lambda col: col.map(_is_empty).mean() * 100)
    cols_to_drop = empty_pct[empty_pct > threshold].index.tolist()

    if verbose:
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns (>{threshold}% empty values):")
            for col in cols_to_drop:
                print(f"  - {col}: {empty_pct[col]:.2f}% empty")
        else:
            print(f"No columns with more than {threshold}% empty values.")
        print(f"\nShape before: {df.shape}")

    df_cleaned = df.drop(columns=cols_to_drop)

    if verbose:
        print(f"Shape after:  {df_cleaned.shape}")

    return df_cleaned


def drop_high_outlier_rows(df: pd.DataFrame, threshold: float, verbose: bool = True) -> pd.DataFrame:
    """
    For each numeric column checks whether the % of outliers (IQR) exceeds threshold.
    If so, removes rows that are outliers in that column.

    Parameters:
        df        : pandas DataFrame
        threshold : threshold in percent (e.g. 30 means above 30% outliers)
        verbose   : whether to print which columns exceeded the threshold
    """
    rows_to_drop = df.index[:0]  # empty Index preserving type
    outlier_info = {}

    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        info = _iqr_outliers(series)
        if info["pct"] > threshold:
            s = series.dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            rows_to_drop = rows_to_drop.union(df.index[outlier_mask])
            outlier_info[col] = info["pct"]

    if verbose:
        if outlier_info:
            print(f"Columns with >{threshold}% outliers — removing their rows:")
            for col, pct in outlier_info.items():
                print(f"  - {col}: {pct:.2f}% outliers")
        else:
            print(f"No columns with more than {threshold}% outliers.")
        print(f"\nShape before: {df.shape}")

    df_cleaned = df.drop(index=rows_to_drop)

    if verbose:
        print(f"Shape after:  {df_cleaned.shape}")

    return df_cleaned

def _is_bool_like(series: pd.Series) -> bool:
    """Returns True for boolean columns: bool dtype, nullable Boolean, and Int64/object with only 0/1."""
    if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.BooleanDtype):
        return True
    return set(series.dropna().unique()) <= {0, 1, True, False}


def drop_high_cardinality_columns(df: pd.DataFrame, threshold: float, verbose: bool = True) -> pd.DataFrame:
    """
    Drops columns (categorical and numeric) where the % of unique values exceeds threshold.

    Cardinality = n_unique / n_non_null * 100. Boolean-like columns are skipped.

    Parameters:
        df        : pandas DataFrame
        threshold : threshold in percent (e.g. 50 means above 50% unique values)
        verbose   : whether to print which columns were dropped
    """
    cols_to_drop = []
    cardinality = {}

    for col in df.columns:
        series = df[col]
        if _is_bool_like(series):
            continue
        n_non_null = series.notna().sum()
        if n_non_null == 0:
            continue
        pct = series.nunique() / n_non_null * 100
        cardinality[col] = pct
        if pct > threshold:
            cols_to_drop.append(col)

    if verbose:
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns (>{threshold}% unique values):")
            for col in cols_to_drop:
                print(f"  - {col}: {cardinality[col]:.2f}% unique")
        else:
            print(f"No columns with more than {threshold}% unique values.")
        print(f"\nShape before: {df.shape}")

    df_cleaned = df.drop(columns=cols_to_drop)

    if verbose:
        print(f"Shape after:  {df_cleaned.shape}")

    return df_cleaned


def drop_low_diversity_columns(df: pd.DataFrame, threshold: float, verbose: bool = True) -> pd.DataFrame:
    """
    Drops columns where a single value dominates more than threshold% of rows (non-null).

    Solves the zero-inflation problem: a column of all zeros (e.g. after outlier removal)
    has 100% dominance of one value and is dropped.

    Parameters:
        df        : pandas DataFrame
        threshold : threshold in percent (e.g. 95 means: drop if top value >= 95% non-null)
        verbose   : whether to print which columns were dropped
    """
    cols_to_drop = []
    dominance = {}

    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        top_pct = series.value_counts().iloc[0] / len(series) * 100
        dominance[col] = top_pct
        if top_pct >= threshold:
            cols_to_drop.append(col)

    if verbose:
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns (one value dominates >={threshold}%):")
            for col in cols_to_drop:
                top_val = df[col].dropna().value_counts().index[0]
                print(f"  - {col}: {dominance[col]:.2f}% of values = {top_val!r}")
        else:
            print(f"No columns with a dominant value >= {threshold}%.")
        print(f"\nShape before: {df.shape}")

    df_cleaned = df.drop(columns=cols_to_drop)

    if verbose:
        print(f"Shape after:  {df_cleaned.shape}")

    return df_cleaned


def rename_columns(df: pd.DataFrame, rename_map: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Renames columns according to the provided mapping dict {old_name: new_name}.

    Parameters:
        df         : pandas DataFrame
        rename_map : mapping dict, e.g. {"AppID": "Title"}
        verbose    : whether to print the changes made
    """
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if verbose:
        if existing:
            for old, new in existing.items():
                print(f"  '{old}' -> '{new}'")
        else:
            print("None of the specified columns exist in the DataFrame.")
    return df.rename(columns=existing)
