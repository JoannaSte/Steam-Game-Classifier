import ast
import pandas as pd


def transform_release_date(df: pd.DataFrame, col: str = "Release date") -> pd.DataFrame:
    """
    Parses the date column and converts it to number of days since the earliest date in the column.

    Earliest date -> 0, later dates -> number of days elapsed since that date.
    Unparseable values -> NaN.

    Parameters:
        df  : pandas DataFrame
        col : name of the column with dates (default "Release date")
    """
    dates = pd.to_datetime(df[col], errors="coerce")
    min_date = dates.min()
    df = df.copy()
    df[col] = (dates - min_date).dt.days.astype("Int64")
    return df


def transform_bool_columns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """
    Converts boolean columns to 0 and 1 (int).

    Parameters:
        df   : pandas DataFrame
        cols : list of columns to convert (None = all bool columns)
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    for col in cols:
        df[col] = df[col].astype("Int64")
    return df


def _parse_lang_list(val) -> list:
    """Parses a string representing a list of languages, e.g. \"['English', 'French']\"."""
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def _transform_language_column(df: pd.DataFrame, col: str, prefix: str, top_n: int = 10) -> pd.DataFrame:
    """
    Decodes a column containing a list of languages into binary columns.

    Removes the original column and adds:
      - dont_have_{prefix}  : 1 if the list is empty [], 0 otherwise
      - {prefix}_{language} : 0/1 for each of the top_n most frequent languages
      - {prefix}_other      : 1 if the row contains at least one language outside the top_n

    Parameters:
        df     : pandas DataFrame
        col    : name of the column with language lists
        prefix : prefix for new columns (e.g. "supported_language")
        top_n  : number of most frequent languages to encode as separate columns
    """
    from collections import Counter

    df = df.copy()
    parsed = df[col].apply(_parse_lang_list)

    counts = Counter(lang for langs in parsed for lang in langs)
    top_langs = {lang for lang, _ in counts.most_common(top_n)}

    df[f"dont_have_{prefix}"] = (parsed.apply(len) == 0).astype("Int64")

    for lang in sorted(top_langs):
        col_name = f"{prefix}_{lang.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = parsed.apply(lambda langs, l=lang: int(l in langs))

    df[f"{prefix}_other"] = parsed.apply(
        lambda langs: int(any(l not in top_langs for l in langs))
    ).astype("Int64")

    df = df.drop(columns=[col])
    return df


def transform_supported_languages(df: pd.DataFrame, col: str = "Supported languages", top_n: int = 10) -> pd.DataFrame:
    """
    Transforms the Supported languages column into binary 0/1 columns.

    Adds dont_have_supported_language, supported_language_{language} for top_n languages
    and supported_language_other for rare languages.
    """
    return _transform_language_column(df, col, "supported_language", top_n=top_n)


def transform_full_audio_languages(df: pd.DataFrame, col: str = "Full audio languages", top_n: int = 10) -> pd.DataFrame:
    """
    Transforms the Full audio languages column into binary 0/1 columns.

    Adds dont_have_full_audio_language, full_audio_language_{language} for top_n languages
    and full_audio_language_other for rare languages.
    """
    return _transform_language_column(df, col, "full_audio_language", top_n=top_n)


def _slugify(val: str) -> str:
    """Converts a string to a safe column name (lowercase, underscores)."""
    return val.lower().strip().replace(" ", "_").replace("-", "_").replace(".", "_")


def _parse_comma_list(val) -> list:
    """Parses a comma-separated string, e.g. 'Valve, Hidden Path'."""
    if pd.isna(val):
        return []
    parts = [p.strip() for p in str(val).split(",") if p.strip()]
    return parts


def _transform_comma_column(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    Decodes a comma-separated column into binary columns.

    Removes the original column and adds:
      - dont_have_{prefix}: 1 if value is NaN, 0 otherwise
      - {prefix}_{value}:   0/1 for each unique value in the data
    """
    df = df.copy()
    parsed = df[col].apply(_parse_comma_list)

    df[f"dont_have_{prefix}"] = (parsed.apply(len) == 0).astype("Int64")

    all_values = sorted({v for vals in parsed for v in vals})
    for val in all_values:
        col_name = f"{prefix}_{_slugify(val)}"
        df[col_name] = parsed.apply(lambda vals, v=val: int(v in vals))

    df = df.drop(columns=[col])
    return df


def transform_comma_columns(df: pd.DataFrame, cols: list, verbose: bool = True) -> pd.DataFrame:
    """
    Transforms multiple comma-separated columns into binary 0/1 columns.

    The prefix for new columns is automatically derived from the column name (slugify).
    For each column adds:
      - dont_have_{prefix}: 1 if value is NaN or empty, 0 otherwise
      - {prefix}_{value}:   0/1 for each unique value in the data

    Parameters:
        df      : pandas DataFrame
        cols    : list of columns to transform, e.g. ["Developers", "Categories", "Genres", "Tags"]
        verbose : whether to print progress
    """
    for col in cols:
        if col not in df.columns:
            if verbose:
                print(f"  Column '{col}' does not exist — skipping.")
            continue
        prefix = _slugify(col)
        if verbose:
            print(f"  Transforming '{col}' -> prefix '{prefix}'")
        df = _transform_comma_column(df, col, prefix)
    return df


def transform_freq_top(
    df: pd.DataFrame,
    cols: list,
    top_n: int = 50,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Transforms comma-separated columns using the frequency + is_top method.

    For each column removes the original and adds two columns:
      - {prefix}_frequency : sum of occurrences of all values in the row across the dataset
                             (e.g. how many times a given developer appears across all games)
      - {prefix}_is_top    : 1 if any value in the row belongs to top_n most frequent, 0 otherwise
      - dont_have_{prefix} : 1 if value is NaN or empty

    Parameters:
        df      : pandas DataFrame
        cols    : list of columns, e.g. ["Developers", "Publishers"]
        top_n   : number of most frequent values defining "top"
        verbose : whether to print progress
    """
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            if verbose:
                print(f"  Column '{col}' does not exist — skipping.")
            continue

        prefix = _slugify(col)
        parsed = df[col].apply(_parse_comma_list)

        # global occurrence counter for each value
        from collections import Counter
        counts = Counter(v for vals in parsed for v in vals)
        top_set = {v for v, _ in counts.most_common(top_n)}

        df[f"dont_have_{prefix}"] = (parsed.apply(len) == 0).astype("Int64")
        df[f"{prefix}_frequency"] = parsed.apply(
            lambda vals: sum(counts[v] for v in vals)
        ).astype("Int64")
        df[f"{prefix}_is_top"] = parsed.apply(
            lambda vals: int(any(v in top_set for v in vals))
        ).astype("Int64")

        df = df.drop(columns=[col])

        if verbose:
            print(f"  '{col}' -> {prefix}_frequency, {prefix}_is_top, dont_have_{prefix}  "
                  f"(top_n={top_n}, unique values={len(counts)})")

    return df


def transform_developers(df: pd.DataFrame, col: str = "Developers") -> pd.DataFrame:
    """Transforms the Developers column into binary 0/1 columns (shortcut for transform_comma_columns)."""
    return _transform_comma_column(df, col, "developer")


def transform_publishers(df: pd.DataFrame, col: str = "Publishers") -> pd.DataFrame:
    """Transforms the Publishers column into binary 0/1 columns (shortcut for transform_comma_columns)."""
    return _transform_comma_column(df, col, "publisher")


def column_types_summary(df: pd.DataFrame):
    """
    Returns two lists: numeric columns and non-numeric columns.

    Parameters:
        df : pandas DataFrame
    Returns:
        (num_cols, cat_cols) — lists of column names
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return num_cols, cat_cols


def transform_owners_range(df: pd.DataFrame, col: str = "Owners range") -> pd.DataFrame:
    """
    Converts owner ranges (e.g. "20000 - 50000") to the midpoint of the interval as int.

    Parameters:
        df  : pandas DataFrame
        col : name of the column with ranges (default "Owners range")
    """
    df = df.copy()
    parts = df[col].astype(str).str.split(r"\s*-\s*")
    lower = pd.to_numeric(parts.str[0].str.strip(), errors="coerce")
    upper = pd.to_numeric(parts.str[1].str.strip(), errors="coerce")
    df[col] = ((lower + upper) / 2).round().astype("Int64")
    return df
