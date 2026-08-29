"""
evaluate_q.py — Valutazione del modello LLM (q_value vs q_llm) con analisi
                di selezione dell'azione ottima
=======================================================================

Adattato per file CSV con colonne:
  id, seed, type, path, file, entry_type, action, step_start, step_end,
  event, n_maps, v_value, q_value, q_llm

Per ogni file CSV:
  - Filtra le righe con entry_type == 1 (valori Q per ogni singola azione).
  - Calcola le metriche di accordo tra q_value (Q* ottima) e q_llm (stima LLM).
  - Valuta se il modello riesce a scegliere l'azione ottima (argmax q_llm == argmax q_value),
    anche quando i valori stimati differiscono dai valori attesi.
  - Genera un set di grafici PNG e scrive un log testuale.
"""

import argparse
import copy
import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr

# ----------------------------------------------------------------------
# Stile grafici e palette
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TYPE_COLORS = {
    "initial": "#1f77b4",
    "worst": "#d62728",
    "intermediate": "#ff7f0e",
    "transition": "#2ca02c",
    "off_track": "#9467bd",
}

ACTION_ORDER = ["left", "right", "forward", "pickup", "toggle"]
ACTION_COLORS = {
    "left":    "#1f77b4",
    "right":   "#ff7f0e",
    "forward": "#2ca02c",
    "pickup":  "#d62728",
    "toggle":  "#9467bd",
}

# Ordine logico preferito per i tipi traiettoria
PREFERRED_TYPE_ORDER = ["initial", "worst", "intermediate", "transition", "off_track"]


def _get_dynamic_type_order(results):
    """Raccoglie dinamicamente i tipi presenti nei risultati mantenendo un ordine logico."""
    found_types = set()
    for r in results:
        if r["metrics"]["mae_by_type"] is not None:
            found_types.update(r["metrics"]["mae_by_type"].index)
        if r["act_summary"]["acc_by_type"] is not None:
            found_types.update(r["act_summary"]["acc_by_type"].index)
    
    # Mantieni l'ordine preferito per i tipi trovati, aggiungi eventuali nuovi tipi alla fine
    ordered = [t for t in PREFERRED_TYPE_ORDER if t in found_types]
    ordered.extend(sorted([t for t in found_types if t not in PREFERRED_TYPE_ORDER]))
    return ordered if ordered else PREFERRED_TYPE_ORDER


# ----------------------------------------------------------------------
# Metriche di accordo (q_value vs q_llm)
# ----------------------------------------------------------------------
def concordance_correlation_coefficient(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx == 0 or sy == 0:
        return float("nan"), float("nan"), float("nan")
    rho = np.corrcoef(x, y)[0, 1]
    ccc = (2 * rho * sx * sy) / (sx**2 + sy**2 + (mx - my) ** 2)
    b = rho * sy / sx
    a = my - b * mx
    return float(ccc), float(b), float(a)


def bootstrap_ci(func, x, y, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    point = func(x, y)
    if n < 5:
        return point, float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            v = func(x[idx], y[idx])
            if np.isfinite(v):
                boots.append(v)
        except Exception:
            continue
    if len(boots) < 50:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def bootstrap_ci_seeded(df, value_col, llm_col, metric, n_boot=500, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    seeds = df["seed"].unique() if "seed" in df.columns else None
    if seeds is None or len(seeds) < 3:
        return bootstrap_ci(
            lambda x, y: {
                "pearson": lambda: pearsonr(x, y)[0],
                "ccc": lambda: concordance_correlation_coefficient(x, y)[0],
                "mae": lambda: np.mean(np.abs(x - y)),
            }[metric](),
            df[value_col].values, df[llm_col].values,
            n_boot=n_boot, seed=seed, alpha=alpha,
        )
    point_data = df[[value_col, llm_col]].values
    x = point_data[:, 0]
    y = point_data[:, 1]

    def f(x_, y_):
        try:
            if metric == "pearson":
                return float(pearsonr(x_, y_)[0])
            if metric == "ccc":
                return float(concordance_correlation_coefficient(x_, y_)[0])
            if metric == "mae":
                return float(np.mean(np.abs(x_ - y_)))
        except Exception:
            return float("nan")
        return float("nan")

    point = f(x, y)
    boots = []
    for _ in range(n_boot):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        parts_x, parts_y = [], []
        for s in sampled_seeds:
            sub = df[df["seed"] == s]
            parts_x.append(sub[value_col].values)
            parts_y.append(sub[llm_col].values)
        if not parts_x:
            continue
        xb = np.concatenate(parts_x)
        yb = np.concatenate(parts_y)
        v = f(xb, yb)
        if np.isfinite(v):
            boots.append(v)
    if len(boots) < 50:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def find_csv_files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))


def label_from_path(csv_path, base_path):
    base_path = base_path if os.path.isdir(base_path) else os.path.dirname(base_path) or "."
    rel = os.path.relpath(csv_path, base_path)
    parts = rel.split(os.sep)
    config = "/".join(parts[:-1]) if len(parts) > 1 else "-"
    fname = os.path.basename(csv_path).replace("output_", "").replace(".csv", "")
    variant = "default"
    model = fname
    if fname.endswith("_no_analisys"):
        model = fname[: -len("_no_analisys")]
        variant = "no_analisys"
    tag = f"{config}_{fname}".replace("/", "_").strip("_")
    tag_label = f"{config} · {fname}" if config != "-" else fname
    return config, model, variant, fname, tag, tag_label


# ----------------------------------------------------------------------
# Metriche valore (q_value vs q_llm su tutte le azioni)
# ----------------------------------------------------------------------
def compute_value_metrics(df_eval, df_raw=None):
    err = df_eval["q_llm"] - df_eval["q_value"]
    abs_err = err.abs()

    try:
        pearson_r, pearson_p = pearsonr(df_eval["q_value"], df_eval["q_llm"])
    except Exception:
        pearson_r, pearson_p = float("nan"), float("nan")
    try:
        spearman_r, spearman_p = spearmanr(df_eval["q_value"], df_eval["q_llm"])
    except Exception:
        spearman_r, spearman_p = float("nan"), float("nan")

    ccc, slope, intercept = concordance_correlation_coefficient(
        df_eval["q_value"].values, df_eval["q_llm"].values)

    x = df_eval["q_value"].values
    y = df_eval["q_llm"].values
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - (intercept + slope * x)) ** 2))
    r2_reg = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mae_by_type = df_eval.assign(abs_err=abs_err).groupby("type")["abs_err"].mean().sort_values(ascending=False)
    mae_by_event = df_eval.assign(abs_err=abs_err).groupby("event")["abs_err"].mean().sort_values(ascending=False)
    mae_by_nmaps = df_eval.assign(abs_err=abs_err).groupby("n_maps")["abs_err"].mean().sort_index() if "n_maps" in df_eval.columns else None

    pearson_ci = bootstrap_ci_seeded(df_eval, "q_value", "q_llm", "pearson", n_boot=500)
    ccc_ci = bootstrap_ci_seeded(df_eval, "q_value", "q_llm", "ccc", n_boot=500)
    mae_ci = bootstrap_ci_seeded(df_eval, "q_value", "q_llm", "mae", n_boot=500)

    if df_raw is not None:
        n_total = len(df_raw)
        n_failures = int((df_raw["q_llm"] == 0.0).sum())
    else:
        n_total = len(df_eval)
        n_failures = 0
    failure_rate = n_failures / n_total if n_total > 0 else float("nan")

    return {
        "n": len(df_eval),
        "n_total": n_total,
        "n_failures": n_failures,
        "failure_rate": failure_rate,
        "mae": abs_err.mean(),
        "mae_ci": mae_ci,
        "rmse": np.sqrt((err**2).mean()),
        "bias": err.mean(),
        "max_err": abs_err.max(),
        "pearson_r": pearson_r, "pearson_p": pearson_p, "pearson_ci": pearson_ci,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "ccc": ccc, "ccc_ci": ccc_ci,
        "slope": slope, "intercept": intercept, "r2_reg": r2_reg,
        "mae_by_type": mae_by_type,
        "mae_by_event": mae_by_event,
        "mae_by_nmaps": mae_by_nmaps,
        "bias_by_type": df_eval.assign(err=err).groupby("type")["err"].mean(),
        "count_by_type": df_eval["type"].value_counts(),
        "count_by_nmaps": df_eval["n_maps"].value_counts().sort_index() if "n_maps" in df_eval.columns else None,
        "n_unique_seeds": int(df_eval["seed"].nunique()) if "seed" in df_eval.columns else None,
    }


# ----------------------------------------------------------------------
# Metriche selezione azione
# ----------------------------------------------------------------------
def compute_action_metrics(df_actions):
    """
    Per ogni stato (gruppo path+seed), calcola:
      - optimal_action = argmax(q_value)
      - llm_action = argmax(q_llm)
      - match = (optimal_action == llm_action)
    """
    rows = []
    for (path, seed), grp in df_actions.groupby(["path", "seed"]):
        if len(grp) == 0:
            continue
        idx_opt = grp["q_value"].idxmax()
        optimal_action = grp.loc[idx_opt, "action"]
        optimal_q_value = grp.loc[idx_opt, "q_value"]

        idx_llm = grp["q_llm"].idxmax()
        llm_action = grp.loc[idx_llm, "action"]
        llm_max_q_llm = grp.loc[idx_llm, "q_llm"]

        opt_row = grp[grp["action"] == optimal_action]
        opt_q_llm = opt_row["q_llm"].values[0] if len(opt_row) > 0 else float("nan")

        llm_chosen_row = grp[grp["action"] == llm_action]
        llm_chosen_q_value = llm_chosen_row["q_value"].values[0] if len(llm_chosen_row) > 0 else float("nan")

        sorted_llm = grp.sort_values("q_llm", ascending=False).reset_index(drop=True)
        rank = (sorted_llm["action"] == optimal_action).idxmax() + 1

        value_loss = optimal_q_value - llm_chosen_q_value
        match = (optimal_action == llm_action)
        is_failure = (grp["q_llm"] == 0.0).all()

        rows.append({
            "path": path,
            "seed": seed,
            "type": grp["type"].iloc[0],
            "event": grp["event"].iloc[0],
            "n_maps": grp["n_maps"].iloc[0],
            "optimal_action": optimal_action,
            "llm_action": llm_action,
            "match": match,
            "rank": int(rank),
            "value_loss": float(value_loss),
            "optimal_q_value": float(optimal_q_value),
            "llm_chosen_q_value": float(llm_chosen_q_value),
            "llm_max_q_llm": float(llm_max_q_llm),
            "optimal_q_llm": float(opt_q_llm),
            "is_failure": bool(is_failure),
        })
    return pd.DataFrame(rows)


def summarize_action_metrics(df_act):
    n = len(df_act)
    n_fail = int(df_act["is_failure"].sum())
    df_valid = df_act[~df_act["is_failure"]].copy()
    n_valid = len(df_valid)

    acc = df_valid["match"].mean() if n_valid > 0 else float("nan")
    top2 = (df_valid["rank"] <= 2).mean() if n_valid > 0 else float("nan")
    top3 = (df_valid["rank"] <= 3).mean() if n_valid > 0 else float("nan")

    acc_by_type = df_valid.groupby("type")["match"].mean().sort_values(ascending=False) if n_valid > 0 else None
    acc_by_event = df_valid.groupby("event")["match"].mean().sort_values(ascending=False) if n_valid > 0 else None
    acc_by_nmaps = df_valid.groupby("n_maps")["match"].mean().sort_index() if n_valid > 0 else None

    mean_value_loss = df_valid["value_loss"].mean() if n_valid > 0 else float("nan")
    cm = pd.crosstab(df_valid["optimal_action"], df_valid["llm_action"]) if n_valid > 0 else None
    per_action = df_valid.groupby("optimal_action")["match"].agg(["mean", "count"]) if n_valid > 0 else None

    return {
        "n_states": n,
        "n_failures": n_fail,
        "n_valid": n_valid,
        "accuracy": acc,
        "top2_accuracy": top2,
        "top3_accuracy": top3,
        "mean_value_loss": mean_value_loss,
        "acc_by_type": acc_by_type,
        "acc_by_event": acc_by_event,
        "acc_by_nmaps": acc_by_nmaps,
        "confusion_matrix": cm,
        "per_action": per_action,
        "df": df_valid,
    }


# ----------------------------------------------------------------------
# Grafici valore (q_value vs q_llm)
# ----------------------------------------------------------------------
def plot_scatter(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for t, color in TYPE_COLORS.items():
        sub = df[df["type"] == t]
        if len(sub):
            ax.scatter(sub["q_value"], sub["q_llm"], s=28, alpha=0.75,
                       label=t, color=color, edgecolor="white", linewidth=0.3)
    # massima leggibilità: assi fissi e identici per tutti i pannelli (Q* in [0,1])
    lo, hi = -0.05, 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="y = x")
    xs = np.linspace(lo, hi, 100)
    ax.fill_between(xs, xs - 0.05, xs + 0.05, color="green", alpha=0.08, label="±0.05")
    ax.fill_between(xs, xs - 0.10, xs + 0.10, color="green", alpha=0.04, label="±0.10")
    if np.isfinite(metrics["slope"]):
        ax.plot(xs, metrics["intercept"] + metrics["slope"] * xs,
                color="red", lw=1.8, alpha=0.9,
                label=f"regressione (slope={metrics['slope']:.2f})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("q_value (Q* ottima)"); ax.set_ylabel("q_llm (stima LLM)")
    ax.set_title(title)
    text = (
        f"Pearson r = {metrics['pearson_r']:.3f}  [{metrics['pearson_ci'][1]:.3f}, {metrics['pearson_ci'][2]:.3f}]\n"
        f"CCC = {metrics['ccc']:.3f}  [{metrics['ccc_ci'][1]:.3f}, {metrics['ccc_ci'][2]:.3f}]\n"
        f"MAE = {metrics['mae']:.4f}  [{metrics['mae_ci'][1]:.4f}, {metrics['mae_ci'][2]:.4f}]\n"
        f"RMSE = {metrics['rmse']:.4f}   bias = {metrics['bias']:+.4f}\n"
        f"slope = {metrics['slope']:.3f}  intercept = {metrics['intercept']:+.3f}\n"
        f"failure rate = {metrics['failure_rate']*100:.1f}%  ({metrics['n_failures']}/{metrics['n_total']})"
    )
    ax.text(0.04, 0.97, text, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_residuals(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    resid = df["q_llm"] - df["q_value"]
    for t, color in TYPE_COLORS.items():
        mask = df["type"] == t
        if mask.sum():
            ax.scatter(df.loc[mask, "q_value"], resid[mask], s=28, alpha=0.75,
                       label=t, color=color, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    ax.axhline(resid.mean(), color="red", linestyle=":", lw=1.6,
               label=f"bias = {resid.mean():+.4f}")
    ax.set_xlabel("q_value"); ax.set_ylabel("residuo (q_llm - q_value)")
    ax.set_title(title); ax.legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_error_hist(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    abs_err = (df["q_value"] - df["q_llm"]).abs()
    bins = np.linspace(0, 0.5, 26)
    ax.hist(abs_err, bins=bins, color="#4c72b0", edgecolor="white", alpha=0.9)
    ax.axvline(abs_err.mean(), color="red", linestyle="--", lw=1.6, label=f"MAE = {abs_err.mean():.4f}")
    ax.axvline(abs_err.median(), color="green", linestyle=":", lw=1.6, label=f"mediana = {abs_err.median():.4f}")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("errore assoluto |q_value - q_llm|"); ax.set_ylabel("frequenza")
    ax.set_title(title); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_bias_by_category(df, category_col, title, path):
    err = df.assign(err=df["q_llm"] - df["q_value"],
                    abs_err=(df["q_value"] - df["q_llm"]).abs())
    grp = err.groupby(category_col)
    bias = grp["err"].mean().sort_values(ascending=False)
    mae = grp["abs_err"].mean()
    counts = df[category_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4c72b0" if b < 0 else "#c44e52" for b in bias.values]
    bars = ax.bar(bias.index.astype(str), bias.values, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, cat in zip(bars, bias.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bias[cat]:+.3f}\nMAE={mae[cat]:.3f}\nn={counts[cat]}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("errore medio (q_llm - q_value)"); ax.set_xlabel(category_col)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_mae_by_category(df, category_col, title, path):
    grp = df.assign(abs_err=(df["q_value"] - df["q_llm"]).abs()).groupby(category_col)["abs_err"].mean().sort_values(ascending=False)
    counts = df[category_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [TYPE_COLORS.get(c, "#888888") for c in grp.index] if category_col == "type" else "#dd8452"
    bars = ax.bar(grp.index.astype(str), grp.values, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, cat in zip(bars, grp.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{grp[cat]:.4f}\nn={counts[cat]}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE"); ax.set_xlabel(category_col); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_mae_by_nmaps(df, title, path):
    if "n_maps" not in df.columns:
        return
    grp = df.assign(abs_err=(df["q_value"] - df["q_llm"]).abs()).groupby("n_maps")["abs_err"].agg(["mean", "std", "count"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = grp.index.astype(int).values
    means = grp["mean"].values
    stds = grp["std"].fillna(0).values
    counts = grp["count"].astype(int).values
    bars = ax.bar(x.astype(str), means, color="#dd8452", edgecolor="black", linewidth=0.6)
    ax.errorbar(x.astype(str), means, yerr=stds / np.sqrt(counts), fmt="none", color="black", capsize=4, lw=1.2)
    for bar, m, n in zip(bars, means, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.4f}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("n_maps (mappe di contesto)"); ax.set_ylabel("MAE"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------
# Grafici selezione azione
# ----------------------------------------------------------------------
def plot_action_accuracy_bar(act_summary, title, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = ["Top-1\n(ottima)", "Top-2", "Top-3"]
    vals = [act_summary["accuracy"], act_summary["top2_accuracy"], act_summary["top3_accuracy"]]
    colors = ["#55a868", "#dd8452", "#4c72b0"]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(1/5, color="gray", linestyle=":", lw=1, label="random (1/5 = 20%)")
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuratezza")
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_action_accuracy_by_category(df_act, category_col, title, path):
    grp = df_act.groupby(category_col)["match"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [TYPE_COLORS.get(c, "#888888") for c in grp.index] if category_col == "type" else "#dd8452"
    bars = ax.bar(grp.index.astype(str), grp["mean"].values, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, (_, row) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{row['mean']*100:.1f}%\nn={int(row['count'])}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1/5, color="gray", linestyle=":", lw=1, label="random (20%)")
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuratezza azione ottima")
    ax.set_xlabel(category_col); ax.set_title(title); ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_confusion_matrix(cm, title, path):
    if cm is None or cm.empty:
        return
    rows = [a for a in ACTION_ORDER if a in cm.index]
    cols = [a for a in ACTION_ORDER if a in cm.columns]
    cm = cm.reindex(index=rows, columns=cols).fillna(0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    row_sums = cm.sum(axis=1).replace(0, 1)
    cm_norm = cm.div(row_sums, axis=0)
    im = ax.imshow(cm_norm.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xlabel("azione scelta dall'LLM"); ax.set_ylabel("azione ottima")
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = cm.values[i, j]
            vn = cm_norm.values[i, j]
            if v > 0:
                col = "white" if vn > 0.5 else "black"
                ax.text(j, i, f"{int(v)}\n({vn*100:.0f}%)",
                        ha="center", va="center", fontsize=8, color=col)
    fig.colorbar(im, ax=ax, label="frazione (per azione ottima)", fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_rank_distribution(df_act, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ranks = df_act["rank"].value_counts().sort_index()
    ranks = ranks.reindex(range(1, 6)).fillna(0)
    colors = ["#55a868", "#a8d08d", "#dd8452", "#c44e52", "#8172b2"]
    bars = ax.bar(ranks.index.astype(str), ranks.values, color=colors,
                  edgecolor="black", linewidth=0.6)
    total = ranks.sum()
    for bar, v in zip(bars, ranks.values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{int(v)}\n({v/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("rank dell'azione ottima nel ranking LLM (1 = migliore)")
    ax.set_ylabel("numero di stati"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_value_loss_distribution(df_act, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    losses = df_act["value_loss"].values
    losses_plot = np.where(losses == 0, 1e-6, losses)
    bins = np.logspace(np.log10(1e-5), np.log10(max(losses.max(), 1e-2)), 30)
    ax.hist(losses_plot, bins=bins, color="#4c72b0", edgecolor="white", alpha=0.9)
    ax.set_xscale("log")
    ax.axvline(losses.mean(), color="red", linestyle="--", lw=1.6,
               label=f"media = {losses.mean():.4f}")
    ax.axvline(np.median(losses), color="green", linestyle=":", lw=1.6,
               label=f"mediana = {np.median(losses):.4f}")
    ax.set_xlabel("value loss (q_value[ottima] - q_value[scelta LLM]), scala log")
    ax.set_ylabel("frequenza"); ax.set_title(title); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_value_error_vs_action_correctness(df_val, df_act, title, path):
    """
    Relazione tra errore sul valore (|q_value - q_llm|) e correttezza dell'azione.
    """
    merged = df_act.merge(
        df_val[["path", "seed", "action", "q_value", "q_llm"]].rename(
            columns={"action": "llm_action", "q_value": "llm_chosen_q_value_check", "q_llm": "llm_chosen_q_llm_check"}
        ),
        on=["path", "seed", "llm_action"], how="inner"
    )
    if merged.empty:
        return

    merged["abs_val_err_llm_choice"] = (merged["llm_chosen_q_value"] - merged["llm_max_q_llm"]).abs()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    correct = merged[merged["match"]]
    wrong = merged[~merged["match"]]
    ax.scatter(correct["abs_val_err_llm_choice"], correct["value_loss"],
               s=30, alpha=0.6, color="#55a868", edgecolor="white", linewidth=0.3,
               label=f"azione corretta (n={len(correct)})")
    ax.scatter(wrong["abs_val_err_llm_choice"], wrong["value_loss"],
               s=30, alpha=0.6, color="#c44e52", edgecolor="white", linewidth=0.3,
               label=f"azione sbagliata (n={len(wrong)})")
    ax.set_xlabel("errore assoluto sul valore della scelta LLM |q_value - q_llm|")
    ax.set_ylabel("value loss (q_value[ottima] - q_value[scelta])")
    ax.set_title("Errore sul valore vs perdita di valore della scelta")
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [correct["abs_val_err_llm_choice"].values, wrong["abs_val_err_llm_choice"].values]
    bp = ax.boxplot(data, tick_labels=["azione corretta", "azione sbagliata"],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black", linewidth=1.6))
    bp["boxes"][0].set_facecolor("#55a868"); bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][1].set_facecolor("#c44e52"); bp["boxes"][1].set_alpha(0.55)
    ax.set_ylabel("errore assoluto sul valore |q_value - q_llm|")
    ax.set_title("Errore sul valore vs correttezza azione")

    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_per_action_accuracy(per_action, title, path):
    if per_action is None or per_action.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    pa = per_action.reindex([a for a in ACTION_ORDER if a in per_action.index])
    colors = [ACTION_COLORS.get(a, "#888888") for a in pa.index]
    bars = ax.bar(pa.index, pa["mean"].values * 100, color=colors,
                  edgecolor="black", linewidth=0.6)
    for bar, (action, row) in zip(bars, pa.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{row['mean']*100:.1f}%\nn={int(row['count'])}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(20, color="gray", linestyle=":", lw=1, label="random (20%)")
    ax.set_ylim(0, 105); ax.set_ylabel("accuratezza (%)")
    ax.set_xlabel("azione ottima"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ----------------------------------------------------------------------
# Confronto globale tra file
# ----------------------------------------------------------------------
def _model_color_map(results):
    models = sorted({r["model"] for r in results})
    palette = plt.get_cmap("tab10").colors
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def _config_style_maps(results):
    configs = sorted({r["config"] for r in results})
    hatches = ["", "///", "...", "xxx", "\\\\\\", "+++"]
    linestyles = ["-", "--", "-.", ":"]
    hatch_map = {c: hatches[i % len(hatches)] for i, c in enumerate(configs)}
    linestyle_map = {c: linestyles[i % len(linestyles)] for i, c in enumerate(configs)}
    return hatch_map, linestyle_map


def plot_global_comparison(results, out_dir, log_lines):
    results = sorted(results, key=lambda r: (r["model"], r["config"], r["variant"]))
    tags = [r["tag_label"] for r in results]
    model_colors = _model_color_map(results)
    config_hatches, config_linestyles = _config_style_maps(results)
    x = np.arange(len(tags))
    w = 0.35

    # Determina l'ordine dei tipi dynamicamente
    type_order = _get_dynamic_type_order(results)

    # --- 00: MAE e RMSE ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    ax.bar(x - w / 2, [r["metrics"]["mae"] for r in results], w, label="MAE", color="#4c72b0")
    ax.bar(x + w / 2, [r["metrics"]["rmse"] for r in results], w, label="RMSE", color="#dd8452")
    mae_errs = np.array([
        [r["metrics"]["mae"] - r["metrics"]["mae_ci"][1], r["metrics"]["mae_ci"][2] - r["metrics"]["mae"]]
        for r in results
    ]).T
    ax.errorbar(x - w / 2, [r["metrics"]["mae"] for r in results],
                yerr=mae_errs, fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("errore"); ax.set_title("MAE (IC 95%) e RMSE — q_value vs q_llm")
    ax.legend(); fig.tight_layout()
    p1 = os.path.join(out_dir, "confronto_00_errori_valore.png"); fig.savefig(p1); plt.close(fig)

    # --- 01: Pearson e CCC ---
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.7), 5))
    pearson_vals = [r["metrics"]["pearson_r"] for r in results]
    pearson_lo = [r["metrics"]["pearson_ci"][1] for r in results]
    pearson_hi = [r["metrics"]["pearson_ci"][2] for r in results]
    ccc_vals = [r["metrics"]["ccc"] for r in results]
    ccc_lo = [r["metrics"]["ccc_ci"][1] for r in results]
    ccc_hi = [r["metrics"]["ccc_ci"][2] for r in results]
    ax.bar(x - w / 2, pearson_vals, w, label="Pearson r", color="#55a868")
    ax.bar(x + w / 2, ccc_vals, w, label="CCC (Lin)", color="#8172b2")
    p_err = np.array([np.array(pearson_vals) - np.array(pearson_lo), np.array(pearson_hi) - np.array(pearson_vals)])
    ax.errorbar(x - w / 2, pearson_vals, yerr=p_err, fmt="none", color="black", capsize=3, lw=1)
    c_err = np.array([np.array(ccc_vals) - np.array(ccc_lo), np.array(ccc_hi) - np.array(ccc_vals)])
    ax.errorbar(x + w / 2, ccc_vals, yerr=c_err, fmt="none", color="black", capsize=3, lw=1)
    ax.axhline(1.0, color="gray", linestyle=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.1); ax.set_ylabel("coefficiente")
    ax.set_title("Pearson r e CCC (Lin) con IC 95% — q_value vs q_llm"); ax.legend()
    fig.tight_layout()
    p2 = os.path.join(out_dir, "confronto_01_correlazioni_valore.png"); fig.savefig(p2); plt.close(fig)

    # --- 02: Boxplot errori valore ---
    tags_n = [f"{t}\n(n={r['metrics']['n']}, fail={r['metrics']['n_failures']})" for t, r in zip(tags, results)]
    fig, ax = plt.subplots(figsize=(max(9, len(tags) * 1.9), 5.8))
    data = [(r["df_val"]["q_value"] - r["df_val"]["q_llm"]).abs().values for r in results]
    data_plot = [np.where(d == 0, 1e-4, d) for d in data]
    bp = ax.boxplot(data_plot, tick_labels=tags_n, showfliers=True, patch_artist=True,
                    widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
                    medianprops=dict(color="black", linewidth=1.6),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="gray", alpha=0.5, markeredgecolor="none"))
    for patch, r in zip(bp["boxes"], results):
        patch.set_facecolor(model_colors[r["model"]]); patch.set_alpha(0.55)
        patch.set_hatch(config_hatches[r["config"]]); patch.set_edgecolor("black")
    ax.set_yscale("log")
    ax.set_ylabel("errore assoluto |q_value - q_llm| (scala log)")
    ax.set_title("Distribuzione errori sui valori Q (entry_type=1)")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8.5)
    model_handles = [Patch(facecolor=model_colors[m], edgecolor="black", alpha=0.55, label=m) for m in sorted(model_colors)]
    config_handles = [Patch(facecolor="white", edgecolor="black", hatch=config_hatches[c], label=c) for c in sorted(config_hatches)]
    leg1 = ax.legend(handles=model_handles, title="modello", loc="lower left", fontsize=7.5, title_fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=config_handles, title="configurazione", loc="lower right", fontsize=7.5, title_fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "confronto_02_boxplot_errori_valore.png"); fig.savefig(p3); plt.close(fig)

    # --- 03: Heatmap MAE per tipo ---
    matrix = np.full((len(type_order), len(results)), np.nan)
    counts_matrix = np.full((len(type_order), len(results)), 0, dtype=int)
    for j, r in enumerate(results):
        for i, t in enumerate(type_order):
            if r["metrics"]["mae_by_type"] is not None and t in r["metrics"]["mae_by_type"].index:
                matrix[i, j] = r["metrics"]["mae_by_type"][t]
                counts_matrix[i, j] = r["metrics"]["count_by_type"].get(t, 0)
    masked = np.ma.masked_invalid(matrix)
    cmap = copy.copy(plt.get_cmap("YlOrRd")); cmap.set_bad(color="#eeeeee")
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.5), max(3.5, len(type_order) * 0.9)))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(type_order))); ax.set_yticklabels(type_order)
    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
    for i in range(len(type_order)):
        for j in range(len(results)):
            val = matrix[i, j]; n = counts_matrix[i, j]
            if np.isnan(val):
                txt = "n/d"; col = "#999999"
            else:
                txt = f"{val:.3f}\nn={n}"; col = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)
    fig.colorbar(im, ax=ax, label="MAE", fraction=0.035, pad=0.02)
    ax.set_title("MAE per tipo traiettoria — q_value vs q_llm"); fig.tight_layout()
    p_heat = os.path.join(out_dir, "confronto_03_heatmap_tipo_valore.png"); fig.savefig(p_heat); plt.close(fig)

    # --- 04: Bias medio ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    biases = [r["metrics"]["bias"] for r in results]
    colors_bias = ["#4c72b0" if b < 0 else "#c44e52" for b in biases]
    ax.bar(x, biases, color=colors_bias, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("bias medio (q_llm - q_value)"); ax.set_title("Bias medio per modello (entry_type=1)")
    fig.tight_layout()
    p_bias = os.path.join(out_dir, "confronto_04_bias_valore.png"); fig.savefig(p_bias); plt.close(fig)

    # --- 05: Failure rate ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    fail_rates = [r["metrics"]["failure_rate"] * 100 for r in results]
    n_failures = [r["metrics"]["n_failures"] for r in results]
    n_totals = [r["metrics"]["n_total"] for r in results]
    colors_f = ["#55a868" if fr < 1 else ("#dd8452" if fr < 5 else "#c44e52") for fr in fail_rates]
    bars = ax.bar(x, fail_rates, color=colors_f, edgecolor="black", linewidth=0.6)
    for bar, fr, nf, nt in zip(bars, fail_rates, n_failures, n_totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{fr:.1f}%\n({nf}/{nt})", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("failure rate (%)"); ax.set_title("Failure rate (q_llm == 0, entry_type=1)")
    fig.tight_layout()
    p_fail = os.path.join(out_dir, "confronto_05_failure_rate_valore.png"); fig.savefig(p_fail); plt.close(fig)

    # --- 06: Calibrazione ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(tags) * 1.5), 5))
    slopes = [r["metrics"]["slope"] for r in results]
    intercepts = [r["metrics"]["intercept"] for r in results]
    ax1.bar(x, slopes, color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax1.axhline(1.0, color="green", linestyle="--", lw=1.5, label="slope ideale = 1")
    ax1.set_xticks(x); ax1.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("slope"); ax1.set_title("Calibrazione: slope"); ax1.legend(fontsize=8)
    ax2.bar(x, intercepts, color="#c44e52", edgecolor="black", linewidth=0.6)
    ax2.axhline(0.0, color="green", linestyle="--", lw=1.5, label="intercept ideale = 0")
    ax2.set_xticks(x); ax2.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("intercept"); ax2.set_title("Calibrazione: intercept"); ax2.legend(fontsize=8)
    fig.suptitle("Calibrazione regressione q_llm ~ q_value", fontsize=11, fontweight="bold")
    fig.tight_layout()
    p_calib = os.path.join(out_dir, "confronto_06_calibrazione_valore.png"); fig.savefig(p_calib); plt.close(fig)

    # --- 07: ECDF errori ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        abs_err = np.sort((r["df_val"]["q_value"] - r["df_val"]["q_llm"]).abs().values)
        y = np.arange(1, len(abs_err) + 1) / len(abs_err)
        ax.plot(abs_err, y, label=r["tag_label"], color=model_colors[r["model"]],
                linestyle=config_linestyles[r["config"]], linewidth=2.2)
    for thr in (0.05, 0.10):
        ax.axvline(thr, color="gray", linestyle=":", linewidth=1)
        ax.text(thr, 0.02, f" soglia {thr}", rotation=90, fontsize=7.5, color="gray", va="bottom")
    ax.set_xlabel("errore assoluto |q_value - q_llm|"); ax.set_ylabel("frazione di stime")
    ax.set_title("ECDF degli errori sui valori Q (entry_type=1)"); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="lower right"); fig.tight_layout()
    p_ecdf = os.path.join(out_dir, "confronto_07_ecdf_errori_valore.png"); fig.savefig(p_ecdf); plt.close(fig)

    # --- 08: Scatter riepilogo ---
    n = len(results); ncols = min(3, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2), squeeze=False)
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=t,
                             markerfacecolor=c, markersize=8) for t, c in TYPE_COLORS.items()]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(TYPE_COLORS),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    # assi fissi per tutti i pannelli — massima leggibilità (Q* in [0,1])
    glo, ghi = -0.05, 1.05
    xs_glo = np.linspace(glo, ghi, 50)
    for i, r in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        df, m = r["df_val"], r["metrics"]
        for t, color in TYPE_COLORS.items():
            sub = df[df["type"] == t]
            if len(sub):
                ax.scatter(sub["q_value"], sub["q_llm"], s=12, alpha=0.7, color=color)
        ax.plot([glo, ghi], [glo, ghi], "k--", lw=1)
        if np.isfinite(m["slope"]):
            ax.plot(xs_glo, m["intercept"] + m["slope"] * xs_glo, color="red", lw=1, alpha=0.8)
        ax.set_xlim(glo, ghi); ax.set_ylim(glo, ghi)
        ax.set_title(f"{r['tag_label']}\nr={m['pearson_r']:.2f} CCC={m['ccc']:.2f} MAE={m['mae']:.3f}", fontsize=9)
        ax.set_xlabel("q_value", fontsize=8); ax.set_ylabel("q_llm", fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle("q_value vs q_llm (entry_type=1)", fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p8 = os.path.join(out_dir, "confronto_08_scatter_riepilogo_valore.png")
    fig.savefig(p8, bbox_inches='tight'); plt.close(fig)

    # --- 09: Action accuracy per modello ---
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.7), 5.5))
    acc_vals = [r["act_summary"]["accuracy"] * 100 for r in results]
    top2_vals = [r["act_summary"]["top2_accuracy"] * 100 for r in results]
    top3_vals = [r["act_summary"]["top3_accuracy"] * 100 for r in results]
    w3 = 0.25
    ax.bar(x - w3, acc_vals, w3, label="Top-1 (ottima)", color="#55a868")
    ax.bar(x, top2_vals, w3, label="Top-2", color="#dd8452")
    ax.bar(x + w3, top3_vals, w3, label="Top-3", color="#4c72b0")
    for i, (a1, a2, a3) in enumerate(zip(acc_vals, top2_vals, top3_vals)):
        ax.text(i - w3, a1 + 1, f"{a1:.0f}%", ha="center", fontsize=7.5)
        ax.text(i, a2 + 1, f"{a2:.0f}%", ha="center", fontsize=7.5)
        ax.text(i + w3, a3 + 1, f"{a3:.0f}%", ha="center", fontsize=7.5)
    ax.axhline(20, color="gray", linestyle=":", lw=1, label="random (20%)")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("accuratezza (%)"); ax.set_ylim(0, 110)
    ax.set_title("Accuratezza selezione azione: argmax(q_llm) == argmax(q_value)?")
    ax.legend(fontsize=8); fig.tight_layout()
    p_act = os.path.join(out_dir, "confronto_09_accuracy_azione.png"); fig.savefig(p_act); plt.close(fig)

    # --- 10: Action accuracy per tipo traiettoria (heatmap) ---
    matrix = np.full((len(type_order), len(results)), np.nan)
    counts_matrix = np.full((len(type_order), len(results)), 0, dtype=int)
    for j, r in enumerate(results):
        if r["act_summary"]["acc_by_type"] is None:
            continue
        for i, t in enumerate(type_order):
            if t in r["act_summary"]["acc_by_type"].index:
                matrix[i, j] = r["act_summary"]["acc_by_type"][t] * 100
                df_t = r["act_summary"]["df"]
                counts_matrix[i, j] = int((df_t["type"] == t).sum())
    masked = np.ma.masked_invalid(matrix)
    cmap = copy.copy(plt.get_cmap("RdYlGn")); cmap.set_bad(color="#eeeeee")
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.5), max(3.5, len(type_order) * 0.9)))
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(type_order))); ax.set_yticklabels(type_order)
    for i in range(len(type_order)):
        for j in range(len(results)):
            val = matrix[i, j]; n = counts_matrix[i, j]
            if np.isnan(val):
                txt = "n/d"; col = "#999999"
            else:
                txt = f"{val:.1f}%\nn={n}"; col = "white" if val < 35 or val > 75 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)
    fig.colorbar(im, ax=ax, label="accuracy (%)", fraction=0.035, pad=0.02)
    ax.set_title("Accuratezza selezione azione per tipo traiettoria"); fig.tight_layout()
    p_act_type = os.path.join(out_dir, "confronto_10_accuracy_per_tipo.png"); fig.savefig(p_act_type); plt.close(fig)

    # --- 11: Confusion matrix per ciascun modello ---
    n_cm = len(results); ncols_cm = min(3, n_cm); nrows_cm = int(np.ceil(n_cm / ncols_cm))
    fig, axes = plt.subplots(nrows_cm, ncols_cm, figsize=(ncols_cm * 5, nrows_cm * 4.8), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncols_cm][i % ncols_cm]
        cm = r["act_summary"]["confusion_matrix"]
        if cm is None or cm.empty:
            ax.text(0.5, 0.5, "n/d", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(r["tag_label"], fontsize=9); continue
        rows = [a for a in ACTION_ORDER if a in cm.index]
        cols = [a for a in ACTION_ORDER if a in cm.columns]
        cm_r = cm.reindex(index=rows, columns=cols).fillna(0)
        row_sums = cm_r.sum(axis=1).replace(0, 1)
        cm_norm = cm_r.div(row_sums, axis=0)
        im = ax.imshow(cm_norm.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
        ax.set_xlabel("LLM", fontsize=8); ax.set_ylabel("ottima", fontsize=8)
        ax.set_title(r["tag_label"], fontsize=9)
        for ii in range(len(rows)):
            for jj in range(len(cols)):
                v = cm_r.values[ii, jj]
                if v > 0:
                    col = "white" if cm_norm.values[ii, jj] > 0.5 else "black"
                    ax.text(jj, ii, f"{int(v)}", ha="center", va="center", fontsize=7, color=col)
    for j in range(n_cm, nrows_cm * ncols_cm):
        axes[j // ncols_cm][j % ncols_cm].axis("off")
    fig.suptitle("Matrice di confusione: azione ottima (righe) vs azione LLM (colonne)", fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p_cm = os.path.join(out_dir, "confronto_11_confusion_matrix.png")
    fig.savefig(p_cm, bbox_inches='tight'); plt.close(fig)

    # --- 12: Value loss medio per modello ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    vloss = [r["act_summary"]["mean_value_loss"] for r in results]
    colors_vl = ["#55a868" if v < 0.01 else ("#dd8452" if v < 0.05 else "#c44e52") for v in vloss]
    bars = ax.bar(x, vloss, color=colors_vl, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vloss):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("value loss medio"); ax.set_title("Value loss: q_value[ottima] - q_value[scelta LLM]")
    fig.tight_layout()
    p_vl = os.path.join(out_dir, "confronto_12_value_loss.png"); fig.savefig(p_vl); plt.close(fig)

    # --- 13: Per-action accuracy ---
    n_pa = len(results); ncols_pa = min(3, n_pa); nrows_pa = int(np.ceil(n_pa / ncols_pa))
    fig, axes = plt.subplots(nrows_pa, ncols_pa, figsize=(ncols_pa * 4.5, nrows_pa * 4), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncols_pa][i % ncols_pa]
        pa = r["act_summary"]["per_action"]
        if pa is None or pa.empty:
            ax.text(0.5, 0.5, "n/d", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(r["tag_label"], fontsize=9); continue
        pa = pa.reindex([a for a in ACTION_ORDER if a in pa.index])
        colors = [ACTION_COLORS.get(a, "#888") for a in pa.index]
        bars = ax.bar(pa.index, pa["mean"].values * 100, color=colors, edgecolor="black", linewidth=0.5)
        for bar, (action, row) in zip(bars, pa.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{row['mean']*100:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(20, color="gray", linestyle=":", lw=0.8)
        ax.set_ylim(0, 110); ax.set_ylabel("accuracy (%)")
        ax.set_title(r["tag_label"], fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=7.5)
    for j in range(n_pa, nrows_pa * ncols_pa):
        axes[j // ncols_pa][j % ncols_pa].axis("off")
    fig.suptitle("Accuratezza per azione ottima (per ogni azione, quante volte l'LLM la sceglie)",
                 fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p_pa = os.path.join(out_dir, "confronto_13_per_action_accuracy.png")
    fig.savefig(p_pa, bbox_inches='tight'); plt.close(fig)

    # --- 14: Action accuracy per n_maps ---
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.5), 5))
    n_maps_vals = sorted({nm for r in results if r["act_summary"]["acc_by_nmaps"] is not None
                          for nm in r["act_summary"]["acc_by_nmaps"].index})
    width = 0.8 / max(len(n_maps_vals), 1)
    for k, nm in enumerate(n_maps_vals):
        vals = []
        for r in results:
            if r["act_summary"]["acc_by_nmaps"] is not None and nm in r["act_summary"]["acc_by_nmaps"].index:
                vals.append(r["act_summary"]["acc_by_nmaps"][nm] * 100)
            else:
                vals.append(0)
        ax.bar(x + k * width - 0.4 + width / 2, vals, width, label=f"n_maps={nm}")
    ax.axhline(20, color="gray", linestyle=":", lw=1, label="random (20%)")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 110)
    ax.set_title("Accuratezza selezione azione per n_maps"); ax.legend(fontsize=8)
    fig.tight_layout()
    p_nm = os.path.join(out_dir, "confronto_14_accuracy_per_nmaps.png"); fig.savefig(p_nm); plt.close(fig)

    # ============================================================
    # Log riassuntivo globale
    # ============================================================
    log_lines.append("\n" + "=" * 70)
    log_lines.append("CONFRONTO GLOBALE TRA MODELLI / CONFIGURAZIONI")
    log_lines.append("=" * 70)

    # Tabella metriche valore
    log_lines.append("\n--- METRICHE VALORE (entry_type=1, q_value vs q_llm) ---")
    header = (f"{'modello/config':42s} {'n':>5s} {'fail%':>6s} {'MAE':>8s} {'RMSE':>8s} "
              f"{'Pearson r [IC95%]':>22s} {'CCC [IC95%]':>22s} {'slope':>6s} {'bias':>8s}")
    log_lines.append(header)
    for r in results:
        m = r["metrics"]
        pearson_str = f"{m['pearson_r']:.3f} [{m['pearson_ci'][1]:.3f},{m['pearson_ci'][2]:.3f}]"
        ccc_str = f"{m['ccc']:.3f} [{m['ccc_ci'][1]:.3f},{m['ccc_ci'][2]:.3f}]"
        log_lines.append(
            f"{r['tag_label']:42s} {m['n']:>5d} {m['failure_rate']*100:>5.1f}% {m['mae']:>8.4f} {m['rmse']:>8.4f} "
            f"{pearson_str:>22s} {ccc_str:>22s} {m['slope']:>6.3f} {m['bias']:+8.4f}"
        )

    # Tabella metriche azione
    log_lines.append("\n--- METRICHE SELEZIONE AZIONE (argmax(q_value) vs argmax(q_llm)) ---")
    header2 = (f"{'modello/config':42s} {'stati':>6s} {'fail':>5s} {'valid':>6s} "
               f"{'Top-1%':>8s} {'Top-2%':>8s} {'Top-3%':>8s} {'VLoss':>8s}")
    log_lines.append(header2)
    for r in results:
        a = r["act_summary"]
        log_lines.append(
            f"{r['tag_label']:42s} {a['n_states']:>6d} {a['n_failures']:>5d} {a['n_valid']:>6d} "
            f"{a['accuracy']*100:>7.1f}% {a['top2_accuracy']*100:>7.1f}% {a['top3_accuracy']*100:>7.1f}% "
            f"{a['mean_value_loss']:>8.4f}"
        )

    # Accuracy per tipo
    log_lines.append("\nAccuracy selezione azione per tipo traiettoria (%):")
    header = f"{'modello/config':42s}" + "".join(f" {t:>10s}" for t in type_order)
    log_lines.append(header)
    for r in results:
        a = r["act_summary"]
        if a["acc_by_type"] is None:
            continue
        row_str = f"{r['tag_label']:42s}"
        for t in type_order:
            if t in a["acc_by_type"].index:
                val = a["acc_by_type"][t] * 100
                row_str += f" {val:>9.1f}%"
            else:
                row_str += f" {'n/d':>10s}"
        log_lines.append(row_str)

    # Per-action accuracy
    log_lines.append("\nAccuracy per azione ottima (per ogni azione ottima, quante volte LLM la sceglie):")
    log_lines.append(f"{'modello/config':42s} {'left':>8s} {'right':>8s} {'forward':>8s} {'pickup':>8s} {'toggle':>8s}")
    for r in results:
        pa = r["act_summary"]["per_action"]
        if pa is None or pa.empty:
            continue
        vals = {a: (pa.loc[a, "mean"] * 100) if a in pa.index else float("nan") for a in ACTION_ORDER}
        log_lines.append(
            f"{r['tag_label']:42s} {vals['left']:>7.1f}% {vals['right']:>7.1f}% "
            f"{vals['forward']:>7.1f}% {vals['pickup']:>7.1f}% {vals['toggle']:>7.1f}%"
        )

    # Giudizio
    log_lines.append("\n" + "-" * 70)
    log_lines.append("GIUDIZIO SUL MODELLO MIGLIORE")
    log_lines.append("-" * 70)
    log_lines.append("Basato su CCC (valore) e accuracy (selezione azione), con IC 95% e failure rate.")
    reliable = [r for r in results if r["metrics"]["n"] >= 50 and r["metrics"]["failure_rate"] < 0.10
                and r["act_summary"]["n_valid"] >= 30]
    if reliable:
        best_val = max(reliable, key=lambda r: r["metrics"]["ccc"])
        best_act = max(reliable, key=lambda r: r["act_summary"]["accuracy"])
        log_lines.append(f"-> Miglior CCC (valore):       {best_val['tag_label']}  CCC={best_val['metrics']['ccc']:.3f}")
        log_lines.append(f"-> Miglior accuracy (azione):  {best_act['tag_label']}  acc={best_act['act_summary']['accuracy']*100:.1f}%")
    else:
        log_lines.append("-> Nessun modello supera le soglie minime (n_val>=50, fail<10%, n_azioni>=30).")

    log_lines.append("")
    log_lines.append("Immagini generate:")
    for p in [p1, p2, p3, p_heat, p_bias, p_fail, p_calib, p_ecdf, p8,
              p_act, p_act_type, p_cm, p_vl, p_pa, p_nm]:
        log_lines.append(f"  - {os.path.basename(p)}")


# ----------------------------------------------------------------------
# Valutazione singolo file
# ----------------------------------------------------------------------
def evaluate_file(csv_path, base_path, charts_dir, log_lines):
    # Legge il CSV auto-detectando il separatore
    df_raw = pd.read_csv(csv_path, sep=None, engine="python", skipinitialspace=True)
    df_raw = df_raw.loc[:, ~df_raw.columns.astype(str).str.contains('^Unnamed')]
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # --- Filtra entry_type == 1 per confrontare q_value e q_llm ---
    if "entry_type" in df_raw.columns:
        df_val_raw = df_raw[df_raw["entry_type"] == 1].copy()
    else:
        # Se la colonna entry_type non esiste, tutte le righe
        # sono considerate entry_type=1 (Q-value per singola azione)
        df_val_raw = df_raw.copy()

    df_val = df_val_raw[df_val_raw["q_llm"] != 0.0].copy()
    n_val_failures = int((df_val_raw["q_llm"] == 0.0).sum())

    if df_val.empty:
        log_lines.append(f"\n[SALTATO] {csv_path}: nessuna stima valida in entry_type=1.")
        return None

    # Calcola metriche sul valore
    value_metrics = compute_value_metrics(df_val, df_raw=df_val_raw)
    
    # Calcola metriche sulla selezione azione
    df_act = compute_action_metrics(df_val_raw) # Passiamo tutto il df per grouping
    act_summary = summarize_action_metrics(df_act)

    config, model, variant, fname, tag, tag_label = label_from_path(csv_path, base_path)
    title = tag_label

    # ============================================================
    # Grafici valore
    # ============================================================
    p_scatter = os.path.join(charts_dir, f"{tag}_1_scatter_valore.png")
    p_resid = os.path.join(charts_dir, f"{tag}_2_residui_valore.png")
    p_hist = os.path.join(charts_dir, f"{tag}_3_distribuzione_errori_valore.png")
    p_type = os.path.join(charts_dir, f"{tag}_4_mae_per_tipo_valore.png")
    p_event = os.path.join(charts_dir, f"{tag}_5_mae_per_evento_valore.png")
    p_bias_type = os.path.join(charts_dir, f"{tag}_6_bias_per_tipo_valore.png")
    p_nmaps = os.path.join(charts_dir, f"{tag}_7_mae_per_nmaps_valore.png")

    plot_scatter(df_val, value_metrics, f"q_value vs q_llm — {title}", p_scatter)
    plot_residuals(df_val, value_metrics, f"Residui valore — {title}", p_resid)
    plot_error_hist(df_val, value_metrics, f"Distribuzione errore valore — {title}", p_hist)
    plot_mae_by_category(df_val, "type", f"MAE per tipo — {title}", p_type)
    plot_mae_by_category(df_val, "event", f"MAE per evento — {title}", p_event)
    plot_bias_by_category(df_val, "type", f"Bias per tipo — {title}", p_bias_type)
    plot_mae_by_nmaps(df_val, f"MAE per n_maps — {title}", p_nmaps)

    # ============================================================
    # Grafici selezione azione
    # ============================================================
    p_act_bar = os.path.join(charts_dir, f"{tag}_8_accuracy_azione.png")
    p_act_type = os.path.join(charts_dir, f"{tag}_9_accuracy_per_tipo_azione.png")
    p_act_event = os.path.join(charts_dir, f"{tag}_10_accuracy_per_evento_azione.png")
    p_cm = os.path.join(charts_dir, f"{tag}_11_confusion_matrix_azione.png")
    p_rank = os.path.join(charts_dir, f"{tag}_12_rank_distribution_azione.png")
    p_vloss = os.path.join(charts_dir, f"{tag}_13_value_loss_azione.png")
    p_err_vs_act = os.path.join(charts_dir, f"{tag}_14_errore_valore_vs_azione.png")
    p_pa = os.path.join(charts_dir, f"{tag}_15_per_action_accuracy.png")

    plot_action_accuracy_bar(act_summary, f"Accuracy selezione azione — {title}", p_act_bar)
    if act_summary["df"] is not None and len(act_summary["df"]) > 0:
        plot_action_accuracy_by_category(act_summary["df"], "type",
                                         f"Accuracy azione per tipo — {title}", p_act_type)
        plot_action_accuracy_by_category(act_summary["df"], "event",
                                         f"Accuracy azione per evento — {title}", p_act_event)
        plot_confusion_matrix(act_summary["confusion_matrix"],
                              f"Matrice di confusione — {title}", p_cm)
        plot_rank_distribution(act_summary["df"],
                               f"Distribuzione rank azione ottima — {title}", p_rank)
        plot_value_loss_distribution(act_summary["df"],
                                     f"Distribuzione value loss — {title}", p_vloss)
        plot_value_error_vs_action_correctness(df_val, act_summary["df"],
                                               f"Errore valore vs correttezza azione — {title}", p_err_vs_act)
        plot_per_action_accuracy(act_summary["per_action"],
                                 f"Accuracy per azione ottima — {title}", p_pa)

    # ============================================================
    # Log
    # ============================================================
    log_lines.append("\n" + "-" * 70)
    log_lines.append(f"FILE: {csv_path}")
    log_lines.append(f"Modello: {model}   |   Configurazione: {config}   |   Variante: {variant}")
    log_lines.append("-" * 70)
    log_lines.append(f"Righe totali nel CSV: {len(df_raw)}")
    log_lines.append(f"  entry_type=1 (q_value vs q_llm): {len(df_val_raw)} righe, di cui {n_val_failures} failure (q_llm==0)")
    log_lines.append(f"  righe valide per metriche (q_llm!=0): {value_metrics['n']}")
    if value_metrics["n_unique_seeds"] is not None:
        log_lines.append(f"  seed unici: {value_metrics['n_unique_seeds']}")

    log_lines.append("")
    log_lines.append("METRICHE VALORE (q_value vs q_llm):")
    log_lines.append(f"  MAE  = {value_metrics['mae']:.4f}  [IC 95%: {value_metrics['mae_ci'][1]:.4f}, {value_metrics['mae_ci'][2]:.4f}]")
    log_lines.append(f"  RMSE = {value_metrics['rmse']:.4f}   Errore max = {value_metrics['max_err']:.4f}")
    log_lines.append(f"  Bias = {value_metrics['bias']:+.4f}  ({'SOVRAstimare' if value_metrics['bias'] > 0 else 'SOTTOstimare'})")
    log_lines.append(f"  Pearson r  = {value_metrics['pearson_r']:.4f}  [IC 95%: {value_metrics['pearson_ci'][1]:.4f}, {value_metrics['pearson_ci'][2]:.4f}]")
    log_lines.append(f"  Spearman rho = {value_metrics['spearman_r']:.4f}")
    log_lines.append(f"  CCC (Lin) = {value_metrics['ccc']:.4f}  [IC 95%: {value_metrics['ccc_ci'][1]:.4f}, {value_metrics['ccc_ci'][2]:.4f}]")
    log_lines.append(f"  R^2 regressione = {value_metrics['r2_reg']:.4f}")
    log_lines.append(f"  Calibrazione: slope={value_metrics['slope']:.4f} (ideale=1), intercept={value_metrics['intercept']:+.4f} (ideale=0)")
    log_lines.append(f"  Failure rate = {value_metrics['failure_rate']*100:.1f}%  ({value_metrics['n_failures']}/{value_metrics['n_total']})")

    log_lines.append("")
    log_lines.append("METRICHE SELEZIONE AZIONE (argmax(q_value) vs argmax(q_llm)):")
    log_lines.append(f"  Top-1 accuracy (azione ottima): {act_summary['accuracy']*100:.1f}%  (su {act_summary['n_valid']} stati validi)")
    log_lines.append(f"  Top-2 accuracy:                 {act_summary['top2_accuracy']*100:.1f}%")
    log_lines.append(f"  Top-3 accuracy:                 {act_summary['top3_accuracy']*100:.1f}%")
    log_lines.append(f"  Value loss medio:               {act_summary['mean_value_loss']:.4f}")
    log_lines.append(f"  (random baseline = 20% per 5 azioni)")

    if act_summary["acc_by_type"] is not None:
        log_lines.append("  Accuracy per tipo traiettoria:")
        for t, v in act_summary["acc_by_type"].items():
            n_t = int((act_summary["df"]["type"] == t).sum()) if act_summary["df"] is not None else 0
            log_lines.append(f"    {t:15s} {v*100:6.1f}%  (n={n_t})")

    if act_summary["per_action"] is not None:
        log_lines.append("  Accuracy per azione ottima:")
        for a, row in act_summary["per_action"].iterrows():
            log_lines.append(f"    {a:10s} {row['mean']*100:6.1f}%  (n={int(row['count'])})")

    if act_summary["confusion_matrix"] is not None:
        log_lines.append("  Matrice di confusione (riga=ottima, colonna=LLM):")
        cm = act_summary["confusion_matrix"]
        cols = [c for c in ACTION_ORDER if c in cm.columns]
        log_lines.append("    " + "  ".join(f"{c:>8s}" for c in cols))
        for idx in cm.index:
            row_vals = "  ".join(f"{int(cm.loc[idx, c]):>8d}" if c in cm.columns else f"{'-':>8s}" for c in cols)
            log_lines.append(f"    {idx:>8s} {row_vals}")

    log_lines.append("")
    log_lines.append("Immagini generate per questo file:")
    for p in [p_scatter, p_resid, p_hist, p_type, p_event, p_bias_type, p_nmaps,
              p_act_bar, p_act_type, p_act_event, p_cm, p_rank, p_vloss, p_err_vs_act, p_pa]:
        if os.path.exists(p):
            log_lines.append(f"  - {os.path.basename(p)}")

    return {
        "df_val": df_val,
        "df_act": df_act,
        "metrics": value_metrics,
        "act_summary": act_summary,
        "tag": tag,
        "tag_label": tag_label,
        "csv_path": csv_path,
        "model": model,
        "config": config,
        "variant": variant,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Valuta q_value vs q_llm e selezione azione.")
    parser.add_argument("--path", type=str, required=True,
                        help="Percorso di un CSV oppure di una cartella (ricerca ricorsiva *.csv)")
    parser.add_argument("--outdir", type=str, default="valutazione_q_output",
                        help="Cartella di output per grafici e log")
    args = parser.parse_args()

    csv_files = find_csv_files(args.path)
    if not csv_files:
        print(f"Nessun file CSV trovato in: {args.path}")
        return

    charts_dir = os.path.join(args.outdir, "grafici")
    os.makedirs(charts_dir, exist_ok=True)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("LOG DI VALUTAZIONE LLM (q_value vs q_llm + selezione azione)")
    log_lines.append(f"Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Sorgente: {args.path}")
    log_lines.append(f"File CSV analizzati: {len(csv_files)}")
    log_lines.append("=" * 70)
    log_lines.append("")
    log_lines.append("NOTE METODOLOGICHE:")
    log_lines.append("- Si processano SOLO righe con entry_type == 1 (5 righe per stato).")
    log_lines.append("- Confronto valori: q_value (Q* ottima per l'azione) vs q_llm (stima LLM).")
    log_lines.append("- Selezione azione: argmax(q_value) = azione ottima, argmax(q_llm) = azione LLM.")
    log_lines.append("- Accuracy = frazione di stati in cui l'LLM sceglie l'azione ottima.")
    log_lines.append("- Value loss = q_value[ottima] - q_value[scelta LLM]: perdita reale di reward.")
    log_lines.append("- Top-K accuracy = l'azione ottima e' nelle prime K posizioni del ranking LLM.")
    log_lines.append("- Failure rate (valore) = righe entry_type=1 con q_llm == 0 (LLM non ha prodotto stima).")

    results = []
    for f in csv_files:
        print(f"Valutazione di: {f}")
        r = evaluate_file(f, args.path, charts_dir, log_lines)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        plot_global_comparison(results, charts_dir, log_lines)
    elif len(results) == 1:
        log_lines.append("\n(Un solo file valutato: nessun confronto globale generato.)")

    log_path = os.path.join(args.outdir, "log_valutazione.txt")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines))

    print("\nCompletato.")
    print(f"Grafici salvati in: {charts_dir}")
    print(f"Log salvato in:     {log_path}")


if __name__ == "__main__":
    main()
