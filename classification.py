import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay, log_loss,
)
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def _save_fig(fig, save_dir, filename):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    safe = ''.join(c if c.isalnum() or c in '-_' else '_' for c in filename)
    path = os.path.join(save_dir, safe + '.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f"  Saved: {path}")


def _clean_names(columns):
    """Cleans column names for LightGBM (replaces special characters with '_')."""
    return [''.join(c if c.isalnum() else '_' for c in col) for col in columns]


def get_full_dataset(df, target_col='is_hit'):
    """
    Returns the full dataset (without splitting or scaling) after removing zero-variance features.
    Used for cross-validation on the entire dataset.

    Returns:
        X             : np.ndarray (unscaled)
        y             : pd.Series
        feature_names : list of column names
    """
    X_raw = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df[target_col].astype(int)
    vt = VarianceThreshold(threshold=0)
    X = vt.fit_transform(X_raw)
    feature_names = X_raw.columns[vt.get_support()].tolist()
    return X, y, feature_names


def prepare_data(df, target_col='is_hit', test_size=0.2, random_state=42):
    """
    Prepares data:
    1. Removes zero-variance features.
    2. Splits into train/test (default 80/20) with stratification.
    3. Fits StandardScaler on the training set (does NOT apply it — data comes out unscaled).

    Data comes out unscaled so that cross_validate_models can scale per fold.
    For training the final model use: scaler.transform(X_train).

    Returns:
        X_train, X_test : np.ndarray (unscaled)
        y_train, y_test : pd.Series
        feature_names   : column names after filtering
        scaler          : StandardScaler fitted on X_train
    """
    X_raw = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df[target_col].astype(int)

    vt = VarianceThreshold(threshold=0)
    X_nv = vt.fit_transform(X_raw)
    feature_names = X_raw.columns[vt.get_support()].tolist()
    X = pd.DataFrame(X_nv, columns=feature_names, index=X_raw.index)

    removed = len(X_raw.columns) - len(feature_names)
    if removed:
        print(f"Removed {removed} zero-variance features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    hit_rate = float(y.mean())
    print(f"Split: train={len(y_train)}, test={len(y_test)}")
    print(f"Class '{target_col}'=1: {hit_rate:.1%}  ({int(y.sum())}/{len(y)})")
    print(f"Features after zero-variance filtering: {len(feature_names)}")

    return X_train.values, X_test.values, y_train, y_test, feature_names, scaler


def _build_models(random_state=42):
    """
    Two models:
    - LogisticRegression with ElasticNet regularization (linear, supports SHAP LinearExplainer)
    - LightGBM (gradient boosting, supports SHAP TreeExplainer)
    Both have class_weight / is_unbalance for the imbalanced 'hit' class.
    """
    return {
        "LogReg (ElasticNet)": LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            is_unbalance=True,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def _build_mlp(random_state=42):
    """Simple MLP network: 256 -> 128 -> 64 -> 1 (ReLU, Adam, L2)."""
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=1,
        warm_start=True,
        random_state=random_state,
    )


def _train_mlp_with_history(X_train, y_train, X_val, y_val, n_epochs=50, random_state=42):
    """
    Trains MLP for n_epochs epochs using partial_fit.
    Handles imbalanced classes via sample_weight.
    Returns trained model with _train_loss and _val_loss attributes.
    """
    mlp = _build_mlp(random_state)
    sample_weights = compute_sample_weight('balanced', y_train)
    classes = np.array([0, 1])
    rng = np.random.default_rng(random_state)

    train_losses, val_losses = [], []
    train_f1s,   val_f1s    = [], []
    for epoch in range(n_epochs):
        idx = rng.permutation(len(X_train))
        y_shuf = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
        mlp.partial_fit(X_train[idx], y_shuf, classes=classes,
                        sample_weight=sample_weights[idx])
        train_losses.append(log_loss(y_train, mlp.predict_proba(X_train)))
        val_losses.append(log_loss(y_val,     mlp.predict_proba(X_val)))
        train_f1s.append(f1_score(y_train, mlp.predict(X_train), zero_division=0))
        val_f1s.append(f1_score(y_val,     mlp.predict(X_val),   zero_division=0))
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}  "
                  f"train_loss={train_losses[-1]:.4f}  val_loss={val_losses[-1]:.4f}  "
                  f"train_f1={train_f1s[-1]:.4f}  val_f1={val_f1s[-1]:.4f}")

    mlp._train_loss = train_losses
    mlp._val_loss   = val_losses
    mlp._train_f1   = train_f1s
    mlp._val_f1     = val_f1s
    return mlp


def cross_validate_models(X, y, feature_names=None, cv=5, random_state=42, n_mlp_epochs=30,
                           plot_fold_history=False, plot_fold_cm=False,
                           plot_shap=False, shap_top_n=15, shap_sample=2000,
                           X_test=None, y_test=None,
                           save_dir=None):
    """
    Cross-validation (StratifiedKFold) on the full dataset.
    Scaling (StandardScaler) happens inside each fold — no data leakage.
    SHAP beeswarm drawn after each fold only for LightGBM (when plot_shap=True).

    Parameters:
        X                : np.ndarray — unscaled features (output of get_full_dataset)
        y                : pd.Series / np.ndarray — labels
        feature_names    : list of feature names — required when plot_shap=True
        plot_fold_history: True = draw MLP training history after each fold
        plot_fold_cm     : True = draw confusion matrices after each fold
        plot_shap        : True = draw SHAP beeswarm for LightGBM after each fold
        shap_top_n       : number of top features on SHAP plot
        shap_sample      : number of val fold samples to use for SHAP
        X_test / y_test  : optional fixed test set — each fold model is evaluated on it

    Returns: pd.DataFrame — rows = models,
             columns = {train|val|test}_{metric}_{mean|std}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    splits = ['train', 'val'] + (['test'] if X_test is not None else [])
    model_names = list(_build_models(random_state).keys()) + ["MLP"]
    fold_scores = {name: {f'{sp}_{m}': [] for sp in splits for m in scoring} for name in model_names}
    classes = np.array([0, 1])

    raw_feat_names = feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    clean_names = _clean_names(raw_feat_names)
    clean_to_orig = dict(zip(clean_names, raw_feat_names))

    def _score(y_true, y_pred, y_prob):
        return {
            'accuracy':  accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall':    recall_score(y_true, y_pred, zero_division=0),
            'f1':        f1_score(y_true, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_true, y_prob),
            'pr_auc':    average_precision_score(y_true, y_prob),
        }

    def _record(name, split, y_true, y_pred, y_prob):
        for m, val in _score(y_true, y_pred, y_prob).items():
            fold_scores[name][f'{split}_{m}'].append(val)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold_idx}/{cv} ---")
        X_f_raw, X_v_raw = X[train_idx], X[val_idx]
        if hasattr(y, 'iloc'):
            y_f, y_v = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_f, y_v = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_f = scaler.fit_transform(X_f_raw)
        X_v = scaler.transform(X_v_raw)
        X_te = scaler.transform(X_test) if X_test is not None else None

        fold_models = {}
        lgbm_model = None

        # --- LogReg and LightGBM ---
        for name, model in _build_models(random_state).items():
            print(f"  [{name}] ...")
            if isinstance(model, lgb.LGBMClassifier):
                X_f_df = pd.DataFrame(X_f, columns=clean_names)
                X_v_df = pd.DataFrame(X_v, columns=clean_names)
                model.fit(X_f_df, y_f)
                _record(name, 'train', y_f, model.predict(X_f_df), model.predict_proba(X_f_df)[:, 1])
                y_pred_v = model.predict(X_v_df)
                _record(name, 'val', y_v, y_pred_v, model.predict_proba(X_v_df)[:, 1])
                if X_te is not None:
                    X_te_df = pd.DataFrame(X_te, columns=clean_names)
                    _record(name, 'test', y_test, model.predict(X_te_df), model.predict_proba(X_te_df)[:, 1])
                lgbm_model = model
            else:
                model.fit(X_f, y_f)
                _record(name, 'train', y_f, model.predict(X_f), model.predict_proba(X_f)[:, 1])
                y_pred_v = model.predict(X_v)
                _record(name, 'val', y_v, y_pred_v, model.predict_proba(X_v)[:, 1])
                if X_te is not None:
                    _record(name, 'test', y_test, model.predict(X_te), model.predict_proba(X_te)[:, 1])
            fold_models[name] = (model, y_pred_v)

        # --- MLP with training history ---
        print(f"  [MLP] ({n_mlp_epochs} epochs) ...")
        sw = compute_sample_weight('balanced', y_f)
        mlp_cv = _build_mlp(random_state)
        rng = np.random.default_rng(random_state)
        train_losses, val_losses, train_f1s, val_f1s = [], [], [], []

        for _ in range(n_mlp_epochs):
            idx = rng.permutation(len(X_f))
            y_shuf = y_f.iloc[idx] if hasattr(y_f, 'iloc') else y_f[idx]
            mlp_cv.partial_fit(X_f[idx], y_shuf, classes=classes, sample_weight=sw[idx])
            if plot_fold_history:
                train_losses.append(log_loss(y_f, mlp_cv.predict_proba(X_f)))
                val_losses.append(log_loss(y_v, mlp_cv.predict_proba(X_v)))
                train_f1s.append(f1_score(y_f, mlp_cv.predict(X_f), zero_division=0))
                val_f1s.append(f1_score(y_v, mlp_cv.predict(X_v), zero_division=0))

        mlp_pred_v = mlp_cv.predict(X_v)
        _record("MLP", 'train', y_f, mlp_cv.predict(X_f), mlp_cv.predict_proba(X_f)[:, 1])
        _record("MLP", 'val',   y_v, mlp_pred_v,          mlp_cv.predict_proba(X_v)[:, 1])
        if X_te is not None:
            _record("MLP", 'test', y_test, mlp_cv.predict(X_te), mlp_cv.predict_proba(X_te)[:, 1])
        fold_models["MLP"] = (mlp_cv, mlp_pred_v)

        # --- MLP training history plot ---
        if plot_fold_history:
            epochs = range(1, n_mlp_epochs + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
            ax1.plot(epochs, train_losses, label='train', color='#1f77b4', lw=1.8)
            ax1.plot(epochs, val_losses,   label='val',   color='#ff7f0e', lw=1.8, linestyle='--')
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Log loss"); ax1.legend(); ax1.grid(alpha=0.3)
            ax1.set_title(f"MLP — loss  |  Fold {fold_idx}/{cv}")
            ax2.plot(epochs, train_f1s, label='train', color='#1f77b4', lw=1.8)
            ax2.plot(epochs, val_f1s,   label='val',   color='#ff7f0e', lw=1.8, linestyle='--')
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 score"); ax2.set_ylim(0, 1.05)
            ax2.legend(); ax2.grid(alpha=0.3)
            ax2.set_title(f"MLP — F1  |  Fold {fold_idx}/{cv}")
            plt.tight_layout()
            _save_fig(fig, save_dir, f'cv_mlp_history_fold{fold_idx}')
            plt.show()
            plt.close(fig)

        # --- Confusion matrices for all models ---
        if plot_fold_cm:
            n_m = len(fold_models)
            fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 4))
            if n_m == 1:
                axes = [axes]
            for ax, (name, (model, y_pred_fold)) in zip(axes, fold_models.items()):
                cm = confusion_matrix(y_v, y_pred_fold)
                ConfusionMatrixDisplay(cm, display_labels=["not_hit", "hit"]).plot(
                    ax=ax, colorbar=False, cmap='Blues'
                )
                ax.set_title(name)
            plt.suptitle(f"Confusion matrices — Fold {fold_idx}/{cv}", fontsize=12, y=1.02)
            plt.tight_layout()
            _save_fig(fig, save_dir, f'cv_confusion_fold{fold_idx}')
            plt.show()
            plt.close(fig)

        # --- SHAP for LightGBM ---
        if plot_shap and lgbm_model is not None:
            try:
                rng_shap = np.random.default_rng(random_state + fold_idx)
                n = X_v.shape[0]
                idx_s = rng_shap.choice(n, size=min(shap_sample, n), replace=False)
                X_shap = pd.DataFrame(X_v[idx_s], columns=clean_names)

                explainer = shap.TreeExplainer(lgbm_model)
                sv_raw = explainer.shap_values(X_shap)
                sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw

                mean_abs = np.abs(sv).mean(axis=0)
                top_idx = np.argsort(mean_abs)[-shap_top_n:][::-1]
                sv_top = sv[:, top_idx]
                X_top = X_shap.iloc[:, top_idx].rename(columns=clean_to_orig)

                fig_shap = plt.figure(figsize=(10, max(6, shap_top_n * 0.35)))
                shap.summary_plot(sv_top, X_top, show=False, max_display=shap_top_n)
                plt.title(f"SHAP — LightGBM  |  Fold {fold_idx}/{cv}  (top {shap_top_n} features)", pad=12)
                plt.tight_layout()
                _save_fig(fig_shap, save_dir, f'cv_shap_fold{fold_idx}')
                plt.show()
                plt.close(fig_shap)
            except Exception as e:
                print(f"  SHAP fold {fold_idx} failed: {e}")

    rows = []
    for name in model_names:
        row = {"model": name}
        for sp in splits:
            for m in scoring:
                arr = np.array(fold_scores[name][f'{sp}_{m}'])
                row[f"{sp}_{m}_mean"] = round(float(arr.mean()), 4)
                row[f"{sp}_{m}_std"]  = round(float(arr.std()),  4)
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def train_models(X_train, y_train, X_val=None, y_val=None,
                 feature_names=None, random_state=42, n_mlp_epochs=50):
    """
    Trains LogReg, LightGBM and MLP on the full training set.
    X_val/y_val are optional — when provided, MLP tracks val_loss each epoch.

    Returns: dict {name: fitted_model}
    """
    models = _build_models(random_state)
    trained = {}

    for name, model in models.items():
        print(f"  Training [{name}] ...")
        if isinstance(model, lgb.LGBMClassifier) and feature_names is not None:
            clean = _clean_names(feature_names)
            model.fit(pd.DataFrame(X_train, columns=clean), y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model

    print(f"  Training [MLP] ({n_mlp_epochs} epochs) ...")
    if X_val is not None and y_val is not None:
        trained["MLP"] = _train_mlp_with_history(
            X_train, y_train, X_val, y_val,
            n_epochs=n_mlp_epochs, random_state=random_state,
        )
    else:
        mlp = _build_mlp(random_state)
        sw = compute_sample_weight('balanced', y_train)
        classes = np.array([0, 1])
        rng = np.random.default_rng(random_state)
        for epoch in range(n_mlp_epochs):
            idx = rng.permutation(len(X_train))
            y_shuf = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
            mlp.partial_fit(X_train[idx], y_shuf, classes=classes, sample_weight=sw[idx])
        trained["MLP"] = mlp

    return trained


def _predict(model, X, feature_names=None):
    """Prediction with name cleaning for LightGBM."""
    if isinstance(model, lgb.LGBMClassifier) and feature_names is not None:
        X_df = pd.DataFrame(X, columns=_clean_names(feature_names))
        return model.predict(X_df), model.predict_proba(X_df)[:, 1]
    return model.predict(X), model.predict_proba(X)[:, 1]


def _metrics_dict(y_true, y_pred, y_prob=None):
    d = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        d["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return d


def evaluate_models(trained_models, X_test, y_test, feature_names=None):
    """
    Evaluates models on the test set.
    Returns: pd.DataFrame with metrics (accuracy, precision, recall, f1, roc_auc) for the test set.
    """
    rows = []
    for name, model in trained_models.items():
        row = {"model": name}
        y_pred, y_prob = _predict(model, X_test, feature_names)
        for metric, val in _metrics_dict(y_test, y_pred, y_prob).items():
            row[f"test_{metric}"] = val
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_cv_comparison(cv_df, splits=None, figsize=None, save_dir=None):
    """
    Bar chart of CV results (mean +/- std).

    Parameters:
        splits: list of splits to plot, e.g. ['val', 'test'] or ['train', 'val', 'test'].
                Defaults to ['val'] if cv_df has no 'test_*' columns, ['val', 'test'] if it does.
                Each split gets its own subplot.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

    if splits is None:
        has_test = any(c.startswith('test_') for c in cv_df.columns)
        splits = ['val', 'test'] if has_test else ['val']

    available = [s for s in splits if any(c.startswith(f'{s}_') for c in cv_df.columns)]
    if not available:
        print("No columns found for the specified splits in cv_df.")
        return

    n_splits = len(available)
    n_models = len(cv_df)
    fig, axes = plt.subplots(n_splits, 1, figsize=figsize or (13, 5 * n_splits), squeeze=False)
    axes = axes[:, 0]

    all_metrics = metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    width = 0.8 / n_models

    for ax, split in zip(axes, available):
        metrics = [m for m in all_metrics if f'{split}_{m}_mean' in cv_df.columns]
        x = np.arange(len(metrics))
        for i, (model_name, row) in enumerate(cv_df.iterrows()):
            means = [row[f"{split}_{m}_mean"] for m in metrics]
            stds  = [row[f"{split}_{m}_std"]  for m in metrics]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                          label=model_name, color=colors[i % len(colors)], capsize=4, alpha=0.85)
            for bar, mean, std in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{mean:.2f}\n±{std:.2f}",
                    ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color='white',
                )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_title(f"CV [{split}] — mean ± std")
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_dir, f'cv_comparison_{"_".join(available)}')
    plt.show()


def plot_training_history(trained_models, figsize=(13, 4), save_dir=None):
    """
    Draws two panels per epoch for MLP models:
      - left:  train loss vs val loss  (log loss / cross-entropy)
      - right: train F1  vs val F1
    Only works when the model has _train_loss / _val_loss / _train_f1 / _val_f1 attributes
    (set by _train_mlp_with_history when X_val/y_val are provided).
    """
    mlp_models = {n: m for n, m in trained_models.items() if hasattr(m, '_train_loss')}
    if not mlp_models:
        print("No MLP models with training history (train_models must receive X_val/y_val).")
        return

    for name, model in mlp_models.items():
        epochs = range(1, len(model._train_loss) + 1)
        has_f1 = hasattr(model, '_train_f1')
        ncols = 2 if has_f1 else 1
        fig, axes = plt.subplots(1, ncols, figsize=(figsize[0] if ncols == 2 else figsize[0] // 2, figsize[1]))
        if ncols == 1:
            axes = [axes]

        # --- loss panel ---
        ax = axes[0]
        ax.plot(epochs, model._train_loss, label='train', color='#1f77b4', lw=1.8)
        if hasattr(model, '_val_loss'):
            ax.plot(epochs, model._val_loss, label='val', color='#ff7f0e', lw=1.8, linestyle='--')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Log loss (cross-entropy)")
        ax.set_title(f"{name} — loss")
        ax.legend()
        ax.grid(alpha=0.3)

        # --- F1 panel ---
        if has_f1:
            ax2 = axes[1]
            ax2.plot(epochs, model._train_f1, label='train', color='#1f77b4', lw=1.8)
            if hasattr(model, '_val_f1'):
                ax2.plot(epochs, model._val_f1, label='val', color='#ff7f0e', lw=1.8, linestyle='--')
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("F1 score")
            ax2.set_title(f"{name} — F1")
            ax2.set_ylim(0, 1.05)
            ax2.legend()
            ax2.grid(alpha=0.3)

        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_')
        _save_fig(fig, save_dir, f'training_history_{safe_name}')
        plt.show()
        plt.close(fig)


def plot_test_metrics(trained_models, X_test, y_test,
                      X_train=None, y_train=None,
                      feature_names=None, figsize=(11, 5), save_dir=None):
    """
    Bar chart of metrics (accuracy, precision, recall, f1) on the test set.
    When X_train/y_train are provided, displays two panels (train | test) for comparison.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    x = np.arange(len(metrics))
    n_models = len(trained_models)
    width = 0.7 / n_models

    def _compute(X, y):
        out = {}
        for name, model in trained_models.items():
            y_pred, _ = _predict(model, X, feature_names)
            out[name] = _metrics_dict(y, y_pred)
        return out

    def _draw_panel(ax, results, title):
        for i, (name, scores) in enumerate(results.items()):
            vals = [scores[m] for m in metrics]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=name,
                          color=colors[i % len(colors)], alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{val:.3f}",
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white',
                )
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    test_results = _compute(X_test, y_test)

    show_train = X_train is not None and y_train is not None
    if show_train:
        train_results = _compute(X_train, y_train)
        fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(figsize[0] * 1.7, figsize[1]))
        _draw_panel(ax_train, train_results, "Metrics — training set")
        _draw_panel(ax_test,  test_results,  "Metrics — test set")
    else:
        fig, ax = plt.subplots(figsize=figsize)
        _draw_panel(ax, test_results, "Metrics on test set")

    plt.tight_layout()
    _save_fig(fig, save_dir, 'test_metrics')
    plt.show()
    plt.close(fig)

    # comparison table
    rows_test = {name: {m: scores[m] for m in metrics} for name, scores in test_results.items()}
    df_test = pd.DataFrame(rows_test).T
    df_test.columns = metric_labels
    print("\nMetrics on test set:")
    print(df_test.to_string())

    if show_train:
        rows_train = {name: {m: scores[m] for m in metrics} for name, scores in train_results.items()}
        df_train = pd.DataFrame(rows_train).T
        df_train.columns = metric_labels
        print("\nMetrics on training set:")
        print(df_train.to_string())
        print("\nDifference (train - test):")
        print((df_train - df_test).to_string())
        return df_test, df_train

    return df_test


def plot_confusion_matrices(trained_models, X_test, y_test, feature_names=None, figsize=(12, 5), save_dir=None):
    """Side-by-side confusion matrices for the test set."""
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, trained_models.items()):
        y_pred, _ = _predict(model, X_test, feature_names)
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=["not_hit", "hit"]).plot(
            ax=ax, colorbar=False, cmap='Blues'
        )
        ax.set_title(name)

    plt.suptitle("Confusion matrices — test set", fontsize=12, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'confusion_matrices')
    plt.show()


def plot_roc_curves(trained_models, X_test, y_test, feature_names=None, figsize=(7, 6), save_dir=None):
    """ROC curves for all models on the test set."""
    colors = ['#1f77b4', '#ff7f0e']
    fig, ax = plt.subplots(figsize=figsize)

    for (name, model), color in zip(trained_models.items(), colors):
        _, y_prob = _predict(model, X_test, feature_names)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  AUC={auc:.3f}")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — test set")
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'roc_curves')
    plt.show()


def plot_shap_importances(trained_models, X_train, feature_names,
                          top_n=20, shap_sample=5000, random_state=42, save_dir=None):
    """
    SHAP beeswarm plots:
    - LogReg: LinearExplainer (positive class)
    - LightGBM: TreeExplainer

    Subsample to shap_sample rows for performance.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X_train.shape[0], size=min(shap_sample, X_train.shape[0]), replace=False)
    X_s = X_train[idx]

    clean_names = _clean_names(feature_names)
    clean_to_orig = dict(zip(clean_names, feature_names))

    for model_name, model in trained_models.items():
        print(f"\n  SHAP: {model_name} ...")
        try:
            if isinstance(model, lgb.LGBMClassifier):
                X_df = pd.DataFrame(X_s, columns=clean_names)
                explainer = shap.TreeExplainer(model)
                sv_raw = explainer.shap_values(X_df)
                sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
                X_disp = X_df.rename(columns=clean_to_orig)
            else:
                explainer = shap.LinearExplainer(model, X_s, feature_names=feature_names)
                sv_raw = explainer.shap_values(X_s)
                sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
                X_disp = pd.DataFrame(X_s, columns=feature_names)

            mean_abs = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-top_n:][::-1]
            sv_top = sv[:, top_idx]
            X_top = X_disp.iloc[:, top_idx]

            plt.figure(figsize=(10, max(6, top_n * 0.35)))
            shap.summary_plot(sv_top, X_top, show=False, max_display=top_n)
            plt.title(f"SHAP — {model_name}  (top {top_n} features)", pad=12)
            plt.tight_layout()
            _save_fig(plt.gcf(), save_dir, f'shap_{model_name}')
            plt.show()
        except Exception as e:
            print(f"  SHAP failed for {model_name}: {e}")


def shap_stability_selection(
    df,
    target_col='is_hit',
    n_iterations=30,
    top_k=20,
    threshold=0.7,
    sample_n=10000,
    random_state=42,
    save_dir=None,
):
    """
    SHAP-based stability selection:
    - n_iterations times draws a bootstrap sample (sample_n rows with replacement)
    - trains LightGBM on each bootstrap
    - computes SHAP and marks top_k features as 'important'
    - returns features that appeared in >= threshold iterations

    This reveals which features are stably important regardless of the specific data split,
    not just important for a single training run.

    Returns:
        stability_scores : pd.Series  — fraction of iterations in which the feature was in top_k
        stable_features  : list[str]  — features with score >= threshold
    """
    X_raw = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df[target_col].astype(int)

    vt = VarianceThreshold(threshold=0)
    X_arr = vt.fit_transform(X_raw)
    feature_names = X_raw.columns[vt.get_support()].tolist()
    clean_names = _clean_names(feature_names)

    counts = pd.Series(0, index=feature_names, dtype=float)
    rng = np.random.default_rng(random_state)

    n_rows = X_arr.shape[0]
    actual_sample = min(sample_n, n_rows)

    for i in range(n_iterations):
        idx = rng.choice(n_rows, size=actual_sample, replace=True)
        X_boot = pd.DataFrame(X_arr[idx], columns=clean_names)
        y_boot = y.iloc[idx].values

        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            random_state=int(random_state) + i,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_boot, y_boot)

        try:
            explainer = shap.TreeExplainer(model)
            sv_raw = explainer.shap_values(X_boot)
            sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
            mean_abs = np.abs(sv).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-top_k:]
            top_features = [feature_names[j] for j in top_idx]
            counts[top_features] += 1
        except Exception:
            pass

        if (i + 1) % 5 == 0:
            print(f"  iteration {i + 1}/{n_iterations} ...")

    stability_scores = (counts / n_iterations).sort_values(ascending=False)
    stable_features = stability_scores[stability_scores >= threshold].index.tolist()

    print(f"\nSelected {len(stable_features)} stable features "
          f"(>= {threshold:.0%} of {n_iterations} iterations).")

    _plot_stability_scores(stability_scores, threshold=threshold, top_k=top_k, save_dir=save_dir)

    return stability_scores, stable_features


def _plot_stability_scores(stability_scores, threshold=0.7, top_k=30, figsize=(9, 8), save_dir=None):
    """Horizontal bar chart of stability selection scores."""
    scores = stability_scores.head(top_k)
    colors = ['#2ca02c' if v >= threshold else '#aec7e8' for v in scores.values]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(scores.index[::-1], scores.values[::-1], color=colors[::-1])
    ax.axvline(threshold, color='red', linestyle='--', lw=1.5,
               label=f'threshold {threshold:.0%}')
    ax.set_xlabel("Stability (% of iterations in top_k)")
    ax.set_title(f"SHAP Stability Selection — top {top_k} features")
    ax.set_xlim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, save_dir, 'stability_scores')
    plt.show()
