"""
evaluate.py — Valutazione del modello LLM (v_value vs v_llm)
==============================================================
Adattato per calcolare le fasce categoriale (Basso/Medio/Alto) usando
soglie relative (quantili 33°/66°) per ogni singolo tipo di traiettoria.
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
# Stile grafici e palette colori (coerente su tutte le immagini)
# ----------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

TYPE_COLORS = {
    "initial": "#1f77b4",
    "worst": "#d62728",
    "intermediate": "#ff7f0e",
    "transition": "#2ca02c",
    "off_track": "#9467bd",
}

# ----------------------------------------------------------------------
# Metriche di accordo
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


def bootstrap_ci_seeded(df, value_col, llm_col, metric, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    seeds = df["seed"].unique() if "seed" in df.columns else None
    if seeds is None or len(seeds) < 3:
        return bootstrap_ci(
            lambda x, y: {
                "pearson": lambda: pearsonr(x, y)[0],
                "ccc": lambda: concordance_correlation_coefficient(x, y)[0],
                "mae": lambda: np.mean(np.abs(x - y)),
            }[metric](),
            df[value_col].values,
            df[llm_col].values,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
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
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    return files


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


def compute_metrics(df_eval, df_raw=None):
    err = df_eval["v_llm"] - df_eval["v_value"]
    abs_err = err.abs()

    try:
        pearson_r, pearson_p = pearsonr(df_eval["v_value"], df_eval["v_llm"])
    except Exception:
        pearson_r, pearson_p = float("nan"), float("nan")
    try:
        spearman_r, spearman_p = spearmanr(df_eval["v_value"], df_eval["v_llm"])
    except Exception:
        spearman_r, spearman_p = float("nan"), float("nan")

    ccc, slope, intercept = concordance_correlation_coefficient(
        df_eval["v_value"].values, df_eval["v_llm"].values
    )

    x = df_eval["v_value"].values
    y = df_eval["v_llm"].values
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - (intercept + slope * x)) ** 2))
    r2_reg = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    mae_by_type = df_eval.assign(abs_err=abs_err).groupby("type")["abs_err"].mean().sort_values(ascending=False)
    mae_by_event = df_eval.assign(abs_err=abs_err).groupby("event")["abs_err"].mean().sort_values(ascending=False)
    mae_by_nmaps = df_eval.assign(abs_err=abs_err).groupby("n_maps")["abs_err"].mean().sort_index()

    pearson_ci = bootstrap_ci_seeded(df_eval, "v_value", "v_llm", "pearson", n_boot=500)
    ccc_ci = bootstrap_ci_seeded(df_eval, "v_value", "v_llm", "ccc", n_boot=500)
    mae_ci = bootstrap_ci_seeded(df_eval, "v_value", "v_llm", "mae", n_boot=500)

    if df_raw is not None:
        n_total = len(df_raw)
        n_failures = int((df_raw["v_llm"] == 0.0).sum())
    else:
        n_total = len(df_eval)
        n_failures = 0
    failure_rate = n_failures / n_total if n_total > 0 else float("nan")

    # --- NUOVO: Calcolo fasce categoriale usando quantili PER OGNI TIPO DI TRAIETTORIA ---
    thresh_method = "quantili 33°/66° per tipo traiettoria"
    thresholds_by_type = {}
    df_parts = []

    for t, sub_df in df_eval.groupby("type"):
        if len(sub_df) >= 3:
            low_thr = float(sub_df["v_value"].quantile(0.33))
            mid_thr = float(sub_df["v_value"].quantile(0.66))
        else:
            # Fallback se ci sono pochissimi dati per quel tipo
            v_min = float(sub_df["v_value"].min())
            v_max = float(sub_df["v_value"].max())
            v_range = v_max - v_min
            low_thr = v_min + v_range / 3.0 if v_range > 0 else v_min
            mid_thr = v_min + 2.0 * v_range / 3.0 if v_range > 0 else v_min
        
        thresholds_by_type[t] = (low_thr, mid_thr)
        
        bins = [-np.inf, low_thr, mid_thr, np.inf]
        labels = ["Low", "Mid", "High"]
        
        sub_df = sub_df.copy()
        sub_df["cat_v_value"] = pd.cut(sub_df["v_value"], bins=bins, labels=labels)
        sub_df["cat_v_llm"] = pd.cut(sub_df["v_llm"], bins=bins, labels=labels)
        df_parts.append(sub_df)

    # Ricostruisce il DataFrame mantenendo l'ordine originale
    if df_parts:
        df_eval = pd.concat(df_parts).sort_index()

    cat_match = (df_eval["cat_v_value"] == df_eval["cat_v_llm"])
    cat_agreement_overall = cat_match.mean() if len(cat_match) > 0 else 0.0
    cat_agreement_by_type = df_eval.assign(match=cat_match).groupby("type")["match"].mean()

    metrics = {
        "n": len(df_eval),
        "n_total": n_total,
        "n_failures": n_failures,
        "failure_rate": failure_rate,
        "mae": abs_err.mean(),
        "mae_ci": mae_ci,
        "rmse": np.sqrt((err**2).mean()),
        "bias": err.mean(),
        "max_err": abs_err.max(),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "pearson_ci": pearson_ci,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "ccc": ccc,
        "ccc_ci": ccc_ci,
        "slope": slope,
        "intercept": intercept,
        "r2_reg": r2_reg,
        "mae_by_type": mae_by_type,
        "mae_by_event": mae_by_event,
        "mae_by_nmaps": mae_by_nmaps,
        "bias_by_type": df_eval.assign(err=err).groupby("type")["err"].mean(),
        "count_by_type": df_eval["type"].value_counts(),
        "count_by_nmaps": df_eval["n_maps"].value_counts().sort_index() if "n_maps" in df_eval.columns else None,
        "n_unique_seeds": int(df_eval["seed"].nunique()) if "seed" in df_eval.columns else None,
        "thresholds": (thresholds_by_type, thresh_method),
        "cat_agreement_overall": float(cat_agreement_overall),
        "cat_agreement_by_type": cat_agreement_by_type,
    }
    
    return df_eval, metrics


# ----------------------------------------------------------------------
# Grafici per singolo file
# ----------------------------------------------------------------------
def plot_scatter(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for t, color in TYPE_COLORS.items():
        sub = df[df["type"] == t]
        if len(sub):
            ax.scatter(sub["v_value"], sub["v_llm"], s=28, alpha=0.75, label=t, color=color, edgecolor="white", linewidth=0.3)
    lo = min(df["v_value"].min(), df["v_llm"].min()) - 0.03
    hi = max(df["v_value"].max(), df["v_llm"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="y = x")
    xs = np.linspace(lo, hi, 100)
    ax.fill_between(xs, xs - 0.05, xs + 0.05, color="green", alpha=0.08, label="±0.05")
    ax.fill_between(xs, xs - 0.10, xs + 0.10, color="green", alpha=0.04, label="±0.10")
    if np.isfinite(metrics["slope"]):
        ax.plot(xs, metrics["intercept"] + metrics["slope"] * xs, color="red", lw=1.8, alpha=0.9,
                label=f"regressione (slope={metrics['slope']:.2f})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("v_value")
    ax.set_ylabel("v_llm")
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
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_residuals(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    resid = df["v_llm"] - df["v_value"]
    for t, color in TYPE_COLORS.items():
        mask = df["type"] == t
        if mask.sum():
            ax.scatter(df.loc[mask, "v_value"], resid[mask], s=28, alpha=0.75, label=t, color=color, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    ax.axhline(resid.mean(), color="red", linestyle=":", lw=1.6, label=f"bias = {resid.mean():+.4f}")
    ax.set_xlabel("v_value")
    ax.set_ylabel("residuo (v_llm - v_value)")
    ax.set_title(title)
    ax.legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_error_hist(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    abs_err = (df["v_value"] - df["v_llm"]).abs()
    bins = np.linspace(0, 0.5, 26)
    ax.hist(abs_err, bins=bins, color="#4c72b0", edgecolor="white", alpha=0.9)
    ax.axvline(abs_err.mean(), color="red", linestyle="--", lw=1.6, label=f"MAE = {abs_err.mean():.4f}")
    ax.axvline(abs_err.median(), color="green", linestyle=":", lw=1.6, label=f"mediana = {abs_err.median():.4f}")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("errore assoluto |v_value - v_llm|")
    ax.set_ylabel("frequenza")
    ax.set_title(title)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bias_by_category(df, category_col, title, path):
    err = df.assign(err=df["v_llm"] - df["v_value"], abs_err=(df["v_value"] - df["v_llm"]).abs())
    grp = err.groupby(category_col)
    bias = grp["err"].mean().sort_values(ascending=False)
    mae = grp["abs_err"].mean()
    counts = df[category_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4c72b0" if b < 0 else "#c44e52" for b in bias.values]
    bars = ax.bar(bias.index.astype(str), bias.values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, cat in zip(bars, bias.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bias[cat]:+.3f}\nMAE={mae[cat]:.3f}\nn={counts[cat]}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("errore medio (v_llm - v_value)")
    ax.set_xlabel(category_col)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mae_by_category(df, category_col, title, path):
    grp = df.assign(abs_err=(df["v_value"] - df["v_llm"]).abs()).groupby(category_col)["abs_err"].mean().sort_values(ascending=False)
    counts = df[category_col].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [TYPE_COLORS.get(c, "#888888") for c in grp.index] if category_col == "type" else "#dd8452"
    bars = ax.bar(grp.index.astype(str), grp.values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, cat in zip(bars, grp.index):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={counts[cat]}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("MAE")
    ax.set_xlabel(category_col)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mae_by_nmaps(df, title, path):
    if "n_maps" not in df.columns:
        return
    grp = df.assign(abs_err=(df["v_value"] - df["v_llm"]).abs()).groupby("n_maps")["abs_err"].agg(["mean", "std", "count"])
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
    ax.set_xlabel("n_maps (mappe di contesto)")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Plot per analisi categoriale (Basso/Medio/Alto) ---
def plot_categorical_agreement(df, metrics, title, path):
    type_order = ["initial", "worst", "intermediate", "transition", "off_track"]
    types_present = [t for t in type_order if t in df["type"].unique()]
    
    n_plots = 1 + len(types_present)
    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    
    # Aumentata l'altezza della figura per fare spazio alla legenda delle soglie
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.3), squeeze=False)
    axes = axes.flatten()
    
    labels = ["Low", "Mid", "High"]
    vmin, vmax = 0, 1

    def plot_heatmap(ax, sub_df, subtitle, thresholds=None):
        if sub_df.empty or "cat_v_value" not in sub_df.columns:
            ax.text(0.5, 0.5, "n/d", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(subtitle, fontsize=10)
            return
            
        conf = pd.crosstab(sub_df["cat_v_value"], sub_df["cat_v_llm"], normalize='index')
        conf = conf.reindex(index=labels, columns=labels).fillna(0).values
        
        im = ax.imshow(conf, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Categoria v_llm")
        ax.set_ylabel("Categoria v_value")
        ax.set_title(subtitle, fontsize=10)

        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if conf[i, j] > 0.5 else "black"
                ax.text(j, i, f"{conf[i, j]*100:.1f}%", ha="center", va="center", color=color, fontsize=9)
        
        # Aggiunge la legenda con i valori numerici delle soglie
        if thresholds:
            low, mid = thresholds
            thresh_str = f"Soglie: Basso ≤ {low:.3f} | Medio ≤ {mid:.3f} | Alto > {mid:.3f}"
            ax.text(0.5, -0.20, thresh_str, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#333333")
            
    thresholds_by_type = metrics.get("thresholds", ({}, ""))[0]
    
    plot_heatmap(axes[0], df, f"Overall\n(Match: {metrics['cat_agreement_overall']*100:.1f}%)")
    
    for i, t in enumerate(types_present):
        sub = df[df["type"] == t]
        match_rate = metrics['cat_agreement_by_type'].get(t, 0.0)
        thrs = thresholds_by_type.get(t, None)
        plot_heatmap(axes[i + 1], sub, f"{t}\n(Match: {match_rate*100:.1f}%)", thresholds=thrs)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Agreement Categorie — {title}", fontweight="bold", y=1.02)
    # Aggiunto rect per fare spazio al testo in basso
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------
# Grafici di confronto globale (tra piu' file/modelli)
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

    # --- MAE e RMSE a confronto ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    ax.bar(x - w / 2, [r["metrics"]["mae"] for r in results], w, label="MAE", color="#4c72b0")
    ax.bar(x + w / 2, [r["metrics"]["rmse"] for r in results], w, label="RMSE", color="#dd8452")
    mae_errs = np.array([
        [r["metrics"]["mae"] - r["metrics"]["mae_ci"][1], r["metrics"]["mae_ci"][2] - r["metrics"]["mae"]]
        for r in results
    ]).T
    ax.errorbar(x - w / 2, [r["metrics"]["mae"] for r in results],
                yerr=mae_errs, fmt="none", color="black", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("errore")
    ax.set_title("MAE (IC 95%) e RMSE per modello e configurazione")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(out_dir, "confronto_00_errori.png")
    fig.savefig(p1)
    plt.close(fig)

    # --- Pearson r e CCC a confronto (entrambi con IC 95%) ---
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
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("coefficiente")
    ax.set_title("Pearson r e CCC (Lin) con IC 95%")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(out_dir, "confronto_01_correlazioni.png")
    fig.savefig(p2)
    plt.close(fig)

    # --- Boxplot (migliorato): scala log con jitter per err=0 ---
    tags_n = [f"{t}\n(n={r['metrics']['n']}, fail={r['metrics']['n_failures']})" for t, r in zip(tags, results)]
    fig, ax = plt.subplots(figsize=(max(9, len(tags) * 1.9), 5.8))
    data = [(r["df"]["v_value"] - r["df"]["v_llm"]).abs().values for r in results]
    data_plot = [np.where(d == 0, 1e-4, d) for d in data]
    bp = ax.boxplot(
        data_plot, tick_labels=tags_n, showfliers=True, patch_artist=True, widths=0.55,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
        medianprops=dict(color="black", linewidth=1.6),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="gray", alpha=0.5, markeredgecolor="none"),
    )
    for patch, r in zip(bp["boxes"], results):
        patch.set_facecolor(model_colors[r["model"]])
        patch.set_alpha(0.55)
        patch.set_hatch(config_hatches[r["config"]])
        patch.set_edgecolor("black")
    ax.set_yscale("log")
    ax.set_ylabel("errore assoluto |v_value - v_llm| (scala log)")
    ax.set_title("Distribuzione degli errori assoluti per modello e configurazione")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8.5)
    model_handles = [Patch(facecolor=model_colors[m], edgecolor="black", alpha=0.55, label=m) for m in sorted(model_colors)]
    config_handles = [Patch(facecolor="white", edgecolor="black", hatch=config_hatches[c], label=c) for c in sorted(config_hatches)]
    leg1 = ax.legend(handles=model_handles, title="modello", loc="lower left", fontsize=7.5, title_fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=config_handles, title="configurazione", loc="lower right", fontsize=7.5, title_fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "confronto_02_boxplot_errori.png")
    fig.savefig(p3)
    plt.close(fig)

    # --- Heatmap: MAE per tipo di traiettoria x modello/configurazione (con n) ---
    type_order = ["initial", "worst", "intermediate", "transition", "off_track"]
    matrix = np.full((len(type_order), len(results)), np.nan)
    counts_matrix = np.full((len(type_order), len(results)), 0, dtype=int)
    for j, r in enumerate(results):
        for i, t in enumerate(type_order):
            if t in r["metrics"]["mae_by_type"].index:
                matrix[i, j] = r["metrics"]["mae_by_type"][t]
                counts_matrix[i, j] = r["metrics"]["count_by_type"].get(t, 0)
    masked = np.ma.masked_invalid(matrix)
    cmap = copy.copy(plt.get_cmap("YlOrRd"))
    cmap.set_bad(color="#eeeeee")
    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.5), 4.8))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(type_order)))
    ax.set_yticklabels(type_order)
    vmax = np.nanmax(matrix)
    for i in range(len(type_order)):
        for j in range(len(results)):
            val = matrix[i, j]
            n = counts_matrix[i, j]
            if np.isnan(val):
                txt = "n/d"
                col = "#999999"
            else:
                txt = f"{val:.3f}\nn={n}"
                col = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col)
    fig.colorbar(im, ax=ax, label="MAE", fraction=0.035, pad=0.02)
    ax.set_title("MAE per tipo di traiettoria e per modello")
    fig.tight_layout()
    p_heat = os.path.join(out_dir, "confronto_03_heatmap_tipo.png")
    fig.savefig(p_heat)
    plt.close(fig)

    # --- Bias medio: il modello sottostima o sovrastima sistematicamente? ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    biases = [r["metrics"]["bias"] for r in results]
    colors_bias = ["#4c72b0" if b < 0 else "#c44e52" for b in biases]
    ax.bar(x, biases, color=colors_bias, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("bias medio (v_llm - v_value)")
    ax.set_title("Bias medio per modello e configurazione")
    fig.tight_layout()
    p_bias = os.path.join(out_dir, "confronto_04_bias.png")
    fig.savefig(p_bias)
    plt.close(fig)

    # --- Bias per tipo di traiettoria: heatmap divergente (con n) ---
    type_order = ["initial", "worst", "intermediate", "transition", "off_track"]
    present_types = [t for t in type_order if any(t in r["metrics"]["bias_by_type"] for r in results)]
    matrix = np.full((len(present_types), len(results)), np.nan)
    for j, r in enumerate(results):
        for i, t in enumerate(present_types):
            matrix[i, j] = r["metrics"]["bias_by_type"].get(t, np.nan)
    masked = np.ma.masked_invalid(matrix)
    cmap = copy.copy(plt.get_cmap("coolwarm"))
    cmap.set_bad(color="#eeeeee")
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.5), 4.8))
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(np.nanmax(np.abs(matrix))) else 1.0
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(present_types)))
    ax.set_yticklabels(present_types)
    for i in range(len(present_types)):
        for j in range(len(results)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/d", ha="center", va="center", fontsize=8, color="#999999")
            else:
                txt = f"{val:+.3f}"
                col = "white" if abs(val) > vmax * 0.75 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    fig.colorbar(im, ax=ax, label="bias (v_llm - v_value)", fraction=0.035, pad=0.02)
    ax.set_title("Bias per tipo di traiettoria e per modello")
    fig.tight_layout()
    p_bias_type = os.path.join(out_dir, "confronto_04b_bias_per_tipo.png")
    fig.savefig(p_bias_type)
    plt.close(fig)

    # --- ECDF: quale quota di stime ha errore sotto una data soglia ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        abs_err = np.sort((r["df"]["v_value"] - r["df"]["v_llm"]).abs().values)
        y = np.arange(1, len(abs_err) + 1) / len(abs_err)
        ax.plot(abs_err, y, label=r["tag_label"], color=model_colors[r["model"]],
                linestyle=config_linestyles[r["config"]], linewidth=2.2)
    for thr in (0.05, 0.10):
        ax.axvline(thr, color="gray", linestyle=":", linewidth=1)
        ax.text(thr, 0.02, f" soglia {thr}", rotation=90, fontsize=7.5, color="gray", va="bottom")
    ax.set_xlabel("errore assoluto |v_value - v_llm|")
    ax.set_ylabel("frazione di stime")
    ax.set_title("ECDF degli errori assoluti")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    p_ecdf = os.path.join(out_dir, "confronto_05_ecdf_errori.png")
    fig.savefig(p_ecdf)
    plt.close(fig)

    # --- Riepilogo scatter (una miniatura per file) con retta y=x + regressione ---
    n = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2), squeeze=False)

    legend_handles = [Line2D([0], [0], marker='o', color='w', label=t,
                             markerfacecolor=c, markersize=8)
                      for t, c in TYPE_COLORS.items()]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(TYPE_COLORS),
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, 1.0))

    for i, r in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        df, m = r["df"], r["metrics"]
        for t, color in TYPE_COLORS.items():
            sub = df[df["type"] == t]
            if len(sub):
                ax.scatter(sub["v_value"], sub["v_llm"], s=12, alpha=0.7, color=color)
        lo = min(df["v_value"].min(), df["v_llm"].min()) - 0.03
        hi = max(df["v_value"].max(), df["v_llm"].max()) + 0.03
        xs = np.linspace(lo, hi, 50)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        if np.isfinite(m["slope"]):
            ax.plot(xs, m["intercept"] + m["slope"] * xs, color="red", lw=1, alpha=0.8, label="regr.")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{r['tag_label']}\nr={m['pearson_r']:.2f}  CCC={m['ccc']:.2f}  MAE={m['mae']:.3f}",
                     fontsize=9)
        ax.set_xlabel("v_value", fontsize=8)
        ax.set_ylabel("v_llm", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("v_value vs v_llm, un grafico per modello", fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p6 = os.path.join(out_dir, "confronto_06_scatter_riepilogo.png")
    fig.savefig(p6, bbox_inches='tight')
    plt.close(fig)

    # --- Failure rate per ogni modello/configurazione ---
    fig, ax = plt.subplots(figsize=(max(7, len(tags) * 1.7), 5))
    fail_rates = [r["metrics"]["failure_rate"] * 100 for r in results]
    n_failures = [r["metrics"]["n_failures"] for r in results]
    n_totals = [r["metrics"]["n_total"] for r in results]
    colors_f = ["#55a868" if fr < 1 else ("#dd8452" if fr < 5 else "#c44e52") for fr in fail_rates]
    bars = ax.bar(x, fail_rates, color=colors_f, edgecolor="black", linewidth=0.6)
    for bar, fr, nf, nt in zip(bars, fail_rates, n_failures, n_totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{fr:.1f}%\n({nf}/{nt})", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("failure rate (%)")
    ax.set_title("Failure rate: stime assenti (v_llm == 0)")
    fig.tight_layout()
    p_fail = os.path.join(out_dir, "confronto_07_failure_rate.png")
    fig.savefig(p_fail)
    plt.close(fig)

    # --- Calibrazione (slope e intercept) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(tags) * 1.5), 5))
    slopes = [r["metrics"]["slope"] for r in results]
    intercepts = [r["metrics"]["intercept"] for r in results]
    ax1.bar(x, slopes, color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax1.axhline(1.0, color="green", linestyle="--", lw=1.5, label="slope ideale = 1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("slope regressione v_llm ~ v_value")
    ax1.set_title("Calibrazione: slope (ideale = 1)")
    ax1.legend(fontsize=8)
    ax2.bar(x, intercepts, color="#c44e52", edgecolor="black", linewidth=0.6)
    ax2.axhline(0.0, color="green", linestyle="--", lw=1.5, label="intercept ideale = 0")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("intercept regressione")
    ax2.set_title("Calibrazione: intercept (ideale = 0)")
    ax2.legend(fontsize=8)
    fig.suptitle("Calibrazione della regressione v_llm ~ v_value", fontsize=11, fontweight="bold")
    fig.tight_layout()
    p_calib = os.path.join(out_dir, "confronto_08_calibrazione.png")
    fig.savefig(p_calib)
    plt.close(fig)

    # --- NUOVO: Confronto Globale Categorical Match Rate ---
    type_order = ["initial", "worst", "intermediate", "transition", "off_track"]
    all_types = [t for t in type_order if any(t in r["metrics"]["cat_agreement_by_type"] for r in results)]

    if all_types:
        n_models = len(results)
        n_types = len(all_types)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(max(8, n_types * 2.5), 6))
        x_pos = np.arange(n_types)

        for i, r in enumerate(results):
            vals = [r["metrics"]["cat_agreement_by_type"].get(t, 0.0) * 100 for t in all_types]
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x_pos + offset, vals, width, label=r["tag_label"],
                          color=model_colors[r["model"]], edgecolor="black", linewidth=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_types, rotation=15, ha="right")
        ax.set_ylabel("Categorical Match Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Confronto Match Categoriale (Basso/Medio/Alto relativo al tipo) per Tipo")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        p_cat_glob = os.path.join(out_dir, "confronto_09_categoriale_globale.png")
        fig.savefig(p_cat_glob)
        plt.close(fig)

    # ============================================================
    # LOG RIASSUNTIVO GLOBALE
    # ============================================================
    log_lines.append("\n" + "=" * 70)
    log_lines.append("CONFRONTO GLOBALE TRA MODELLI / CONFIGURAZIONI")
    log_lines.append("=" * 70)
    header = (
        f"{'modello/config':42s} {'n':>5s} {'fail%':>6s} {'MAE':>8s} {'RMSE':>8s} "
        f"{'Pearson r [IC95%]':>22s} {'CCC [IC95%]':>22s} {'slope':>6s} {'bias':>8s}"
    )
    log_lines.append(header)
    for r in results:
        m = r["metrics"]
        pearson_str = f"{m['pearson_r']:.3f} [{m['pearson_ci'][1]:.3f},{m['pearson_ci'][2]:.3f}]"
        ccc_str = f"{m['ccc']:.3f} [{m['ccc_ci'][1]:.3f},{m['ccc_ci'][2]:.3f}]"
        log_lines.append(
            f"{r['tag_label']:42s} {m['n']:>5d} {m['failure_rate']*100:>5.1f}% {m['mae']:>8.4f} {m['rmse']:>8.4f} "
            f"{pearson_str:>22s} {ccc_str:>22s} {m['slope']:>6.3f} {m['bias']:+8.4f}"
        )

    log_lines.append("\nMAE per n_maps (numero di mappe di contesto):")
    log_lines.append(f"{'modello/config':42s} {'n_maps=1':>10s} {'n_maps=3':>10s} {'n_maps=5':>10s}")
    for r in results:
        m = r["metrics"]
        if m["mae_by_nmaps"] is None:
            continue
        v1 = m["mae_by_nmaps"].get(1, float("nan"))
        v3 = m["mae_by_nmaps"].get(3, float("nan"))
        v5 = m["mae_by_nmaps"].get(5, float("nan"))
        log_lines.append(
            f"{r['tag_label']:42s} {v1:>10.4f} {v3:>10.4f} {v5:>10.4f}"
        )

    log_lines.append("")
    log_lines.append("-" * 70)
    log_lines.append("GIUDIZIO SUL MODELLO MIGLIORE")
    log_lines.append("-" * 70)
    log_lines.append("NOTA METODOLOGICA: il giudizio non si basa solo sul MAE, ma su un insieme")
    log_lines.append("di metriche che includono IC 95%, failure rate, CCC (calibrazione) e sample size.")
    log_lines.append("Modelli con sample size molto piccoli (n < 100) hanno IC larghi e non sono")
    log_lines.append("direttamente confrontabili con quelli con n > 300. In caso di IC sovrapposti,")
    log_lines.append("le differenze non sono statisticamente significative.")
    log_lines.append("")

    reliable_idx = set()
    for i, r in enumerate(results):
        if r["metrics"]["n"] >= 100 and r["metrics"]["failure_rate"] < 0.05:
            reliable_idx.add(i)
    reliable = [results[i] for i in sorted(reliable_idx)]
    if reliable:
        best_ccc = max(reliable, key=lambda r: r["metrics"]["ccc"])
        worst_ccc = min(reliable, key=lambda r: r["metrics"]["ccc"])
        log_lines.append(f"-> Miglior CCC (calibrazione + accordo): {best_ccc['tag_label']}")
        log_lines.append(f"   CCC = {best_ccc['metrics']['ccc']:.3f} [IC 95%: {best_ccc['metrics']['ccc_ci'][1]:.3f}, {best_ccc['metrics']['ccc_ci'][2]:.3f}]")
        log_lines.append(f"   MAE = {best_ccc['metrics']['mae']:.4f} [IC 95%: {best_ccc['metrics']['mae_ci'][1]:.4f}, {best_ccc['metrics']['mae_ci'][2]:.4f}]")
        log_lines.append(f"   failure rate = {best_ccc['metrics']['failure_rate']*100:.1f}%, slope = {best_ccc['metrics']['slope']:.3f}")
        log_lines.append("")
        log_lines.append(f"-> Peggiore CCC: {worst_ccc['tag_label']}")
        log_lines.append(f"   CCC = {worst_ccc['metrics']['ccc']:.3f} [IC 95%: {worst_ccc['metrics']['ccc_ci'][1]:.3f}, {worst_ccc['metrics']['ccc_ci'][2]:.3f}]")
    else:
        log_lines.append("-> Nessun modello supera la soglia di affidabilita' (n >= 100 AND failure rate < 5%).")

    excluded = [r for i, r in enumerate(results) if i not in reliable_idx]
    if excluded:
        log_lines.append("")
        log_lines.append("Modelli esclusi dal giudizio (n < 100 oppure failure rate >= 5%):")
        for r in excluded:
            m = r["metrics"]
            reason = []
            if m["n"] < 100:
                reason.append(f"n={m['n']} (<100)")
            if m["failure_rate"] >= 0.05:
                reason.append(f"failure rate={m['failure_rate']*100:.1f}% (>=5%)")
            log_lines.append(f"  - {r['tag_label']}: {', '.join(reason)}")

    log_lines.append("")
    log_lines.append("Percentuale di stime con errore sotto una soglia:")
    log_lines.append(f"{'modello/config':42s} {'entro 0.05':>12s} {'entro 0.10':>12s}")
    for r in results:
        abs_err = (r["df"]["v_value"] - r["df"]["v_llm"]).abs()
        pct05 = (abs_err <= 0.05).mean() * 100
        pct10 = (abs_err <= 0.10).mean() * 100
        log_lines.append(f"{r['tag_label']:42s} {pct05:>11.1f}% {pct10:>11.1f}%")

    # --- NUOVO LOG CATEGORIALE GLOBALE ---
    log_lines.append("")
    log_lines.append("Accordo Categoriale (Basso/Medio/Alto basato su quantili 33°/66° per tipo traiettoria):")
    log_lines.append(f"{'modello/config':42s} {'Overall Match':>15s} {'initial':>10s} {'worst':>10s} {'interm.':>10s} {'trans.':>10s} {'off_tr.':>10s}")
    for r in results:
        m = r["metrics"]
        ovr = m["cat_agreement_overall"] * 100
        vals = [m["cat_agreement_by_type"].get(t, 0.0) * 100 for t in ["initial", "worst", "intermediate", "transition", "off_track"]]
        log_lines.append(f"{r['tag_label']:42s} {ovr:>14.1f}% {vals[0]:>9.1f}% {vals[1]:>9.1f}% {vals[2]:>9.1f}% {vals[3]:>9.1f}% {vals[4]:>9.1f}%")

    log_lines.append("")
    log_lines.append("Immagini di confronto generate:")
    log_lines.append(f"  - {os.path.basename(p1)}: MAE (con IC 95%) e RMSE per ogni modello/config.")
    log_lines.append(f"  - {os.path.basename(p2)}: Pearson r e CCC (Lin) con IC 95% per ogni modello/config.")
    log_lines.append(f"  - {os.path.basename(p3)}: boxplot (scala log, jitter per err=0) degli errori assoluti; colore = modello, tratteggio = config.")
    log_lines.append(f"  - {os.path.basename(p_heat)}: MAE per tipo di traiettoria (con n), una colonna per modello/config.")
    log_lines.append(f"  - {os.path.basename(p_bias)}: bias medio con segno; >0 = sovrastima, <0 = sottostima.")
    log_lines.append(f"  - {os.path.basename(p_bias_type)}: heatmap del bias per tipo di traiettoria; blu = sottostima, rosso = sovrastima.")
    log_lines.append(f"  - {os.path.basename(p_ecdf)}: ECDF degli errori assoluti; la quota di stime sotto una data soglia.")
    log_lines.append(f"  - {os.path.basename(p6)}: griglia di scatter (con retta y=x e retta di regressione), uno per modello/config.")
    log_lines.append(f"  - {os.path.basename(p_fail)}: failure rate per ogni modello/config (verde < 1%, arancio < 5%, rosso >= 5%).")
    log_lines.append(f"  - {os.path.basename(p_calib)}: calibrazione (slope e intercept della regressione v_llm ~ v_value).")
    if all_types:
        log_lines.append(f"  - {os.path.basename(p_cat_glob)}: confronto del match categoriale (Basso/Medio/Alto relativo al tipo) diviso per tipo di traiettoria.")


# ----------------------------------------------------------------------
# Valutazione di un singolo file
# ----------------------------------------------------------------------
def evaluate_file(csv_path, base_path, charts_dir, log_lines):
    df_raw = pd.read_csv(csv_path)
    df_eval = df_raw[df_raw["v_llm"] != 0.0].copy()
    n_excluded = len(df_raw) - len(df_eval)
    n_failures = int((df_raw["v_llm"] == 0.0).sum())

    if df_eval.empty:
        log_lines.append(f"\n[SALTATO] {csv_path}: nessuna stima valida ({n_failures} failure su {len(df_raw)} righe).")
        return None

    # Ora compute_metrics restituisce il df aggiornato con le colonne categoriale e le metriche
    df_eval, metrics = compute_metrics(df_eval, df_raw=df_raw)
    
    config, model, variant, fname, tag, tag_label = label_from_path(csv_path, base_path)
    title = f"{tag_label}"

    p_scatter = os.path.join(charts_dir, f"{tag}_1_scatter.png")
    p_resid = os.path.join(charts_dir, f"{tag}_2_residui.png")
    p_hist = os.path.join(charts_dir, f"{tag}_3_distribuzione_errori.png")
    p_type = os.path.join(charts_dir, f"{tag}_4_mae_per_tipo.png")
    p_event = os.path.join(charts_dir, f"{tag}_5_mae_per_evento.png")
    p_bias_type = os.path.join(charts_dir, f"{tag}_6_bias_per_tipo.png")
    p_nmaps = os.path.join(charts_dir, f"{tag}_7_mae_per_nmaps.png")
    p_cat = os.path.join(charts_dir, f"{tag}_8_categoriale.png")

    plot_scatter(df_eval, metrics, f"v_value vs v_llm — {title}", p_scatter)
    plot_residuals(df_eval, metrics, f"Residui — {title}", p_resid)
    plot_error_hist(df_eval, metrics, f"Distribuzione errore assoluto — {title}", p_hist)
    plot_mae_by_category(df_eval, "type", f"MAE per tipo di traiettoria — {title}", p_type)
    plot_mae_by_category(df_eval, "event", f"MAE per evento — {title}", p_event)
    plot_bias_by_category(df_eval, "type", f"Bias per tipo di traiettoria — {title}", p_bias_type)
    plot_mae_by_nmaps(df_eval, f"MAE per n_maps (contesto fornito all'LLM) — {title}", p_nmaps)
    plot_categorical_agreement(df_eval, metrics, title, p_cat)

    log_lines.append("\n" + "-" * 70)
    log_lines.append(f"FILE: {csv_path}")
    log_lines.append(f"Modello: {model}   |   Configurazione: {config}   |   Variante: {variant}")
    log_lines.append("-" * 70)
    log_lines.append(f"Righe totali nel CSV: {len(df_raw)}")
    log_lines.append(f"  di cui FAILURE (v_llm == 0.0): {n_failures}  ({metrics['failure_rate']*100:.1f}%)")
    log_lines.append(f"  di cui valutate (v_llm != 0.0): {metrics['n']}")
    if metrics["n_unique_seeds"] is not None:
        log_lines.append(f"  seed unici rappresentati: {metrics['n_unique_seeds']}")
    log_lines.append("")
    log_lines.append("Metriche di accordo (calcolate solo sulle stime valide):")
    log_lines.append(f"  MAE  (errore assoluto medio):        {metrics['mae']:.4f}  [IC 95%: {metrics['mae_ci'][1]:.4f}, {metrics['mae_ci'][2]:.4f}]")
    log_lines.append(f"  RMSE (radice errore quadratico):     {metrics['rmse']:.4f}")
    log_lines.append(f"  Errore massimo:                      {metrics['max_err']:.4f}")
    log_lines.append(f"  Bias medio (v_llm - v_value):        {metrics['bias']:+.4f}  "
                      f"({'v_llm tende a SOVRAstimare' if metrics['bias'] > 0 else 'v_llm tende a SOTTOstimare'})")
    log_lines.append(f"  Pearson r  (correlazione lineare):   {metrics['pearson_r']:.4f}  [IC 95%: {metrics['pearson_ci'][1]:.4f}, {metrics['pearson_ci'][2]:.4f}]  (p-value={metrics['pearson_p']:.2e})")
    log_lines.append(f"  Spearman rho (correlazione di rango):{metrics['spearman_r']:.4f}  (p-value={metrics['spearman_p']:.2e})")
    log_lines.append(f"  CCC (Lin, concordanza):               {metrics['ccc']:.4f}  [IC 95%: {metrics['ccc_ci'][1]:.4f}, {metrics['ccc_ci'][2]:.4f}]")
    log_lines.append(f"  R^2 di regressione (vero, NON pearson^2): {metrics['r2_reg']:.4f}")
    log_lines.append(f"  Calibrazione:  slope = {metrics['slope']:.4f},  intercept = {metrics['intercept']:+.4f}")
    
    # --- NUOVO LOG CATEGORIALE SINGOLO ---
    log_lines.append("")
    log_lines.append("Accordo Categoriale (Basso/Medio/Alto relativo al tipo):")
    thresholds_by_type, thresh_method = metrics["thresholds"]
    log_lines.append(f"  Metodo soglie: {thresh_method}")
    for t, (low_thr, mid_thr) in thresholds_by_type.items():
        log_lines.append(f"    Tipo '{t}': Basso <= {low_thr:.3f} <= Medio <= {mid_thr:.3f} <= Alto")
    log_lines.append(f"  Match Overall (v_llm nella stessa fascia di v_value): {metrics['cat_agreement_overall']*100:.1f}%")
    for t, match_rate in metrics["cat_agreement_by_type"].items():
        log_lines.append(f"    Tipo '{t}': {match_rate*100:.1f}%")

    log_lines.append("")
    log_lines.append("MAE per tipo di traiettoria:")
    for t, v in metrics["mae_by_type"].items():
        log_lines.append(f"    {t:15s} MAE={v:.4f}  (n={metrics['count_by_type'][t]})")
    log_lines.append("MAE per evento:")
    for e, v in metrics["mae_by_event"].items():
        log_lines.append(f"    {e:15s} MAE={v:.4f}")
    if metrics["mae_by_nmaps"] is not None and len(metrics["mae_by_nmaps"]):
        log_lines.append("MAE per n_maps (contesto fornito all'LLM):")
        for nm, v in metrics["mae_by_nmaps"].items():
            n_count = metrics["count_by_nmaps"].get(nm, 0) if metrics["count_by_nmaps"] is not None else 0
            log_lines.append(f"    n_maps={nm}    MAE={v:.4f}  (n={n_count})")
    
    log_lines.append("Immagini generate per questo file:")
    log_lines.append(f"  - {os.path.basename(p_scatter)}")
    log_lines.append(f"  - {os.path.basename(p_resid)}")
    log_lines.append(f"  - {os.path.basename(p_hist)}")
    log_lines.append(f"  - {os.path.basename(p_type)}")
    log_lines.append(f"  - {os.path.basename(p_event)}")
    log_lines.append(f"  - {os.path.basename(p_bias_type)}")
    log_lines.append(f"  - {os.path.basename(p_nmaps)}")
    log_lines.append(f"  - {os.path.basename(p_cat)}: heatmaps 3x3 dell'accordo categoriale (Overall + per tipo).")

    return {
        "df": df_eval,
        "df_raw": df_raw,
        "metrics": metrics,
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
    parser = argparse.ArgumentParser(description="Valuta l'accordo tra v_value e v_llm, genera grafici e un log riassuntivo.")
    parser.add_argument("--path", type=str, required=True, help="Percorso di un CSV oppure di una cartella (ricerca ricorsiva di *.csv)")
    parser.add_argument("--outdir", type=str, default="valutazione_output", help="Cartella di output per grafici e log")
    args = parser.parse_args()

    csv_files = find_csv_files(args.path)
    if not csv_files:
        print(f"Nessun file CSV trovato in: {args.path}")
        return

    charts_dir = os.path.join(args.outdir, "grafici")
    os.makedirs(charts_dir, exist_ok=True)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("LOG DI VALUTAZIONE DEL MODELLO LLM (v_value vs v_llm)")
    log_lines.append(f"Generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append(f"Sorgente: {args.path}")
    log_lines.append(f"File CSV analizzati: {len(csv_files)}")
    log_lines.append("=" * 70)
    log_lines.append("")
    log_lines.append("NOTE METODOLOGICHE:")
    log_lines.append("- v_value e' la V* ottima calcolata via value iteration (gamma=0.99).")
    log_lines.append("- Le righe con v_llm == 0.0 vengono contate come FAILURE e riportate come failure_rate.")
    log_lines.append("- Pearson r e CCC (Lin) sono accompagnati da IC 95% via bootstrap sui seed.")
    log_lines.append("- CCC e' la metrica corretta per misurare ACCORDO: penalizza bias sistematici e slope != 1.")
    log_lines.append("- R^2 ora e' quello di regressione (NON piu' Pearson^2).")
    log_lines.append("- La calibrazione (slope, intercept di v_llm ~ v_value) e' riportata esplicitamente.")
    log_lines.append("- MAE stratificato per n_maps permette di confrontare i tipi di traiettoria a parita' di contesto.")
    log_lines.append("- Il giudizio 'migliore' non si basa solo sul MAE, ma su IC, CCC, failure rate e sample size.")
    log_lines.append("- L'AGREEMENT CATEGORIALE valuta se le stime cadono nelle fasce Basso/Medio/Alto.")
    log_lines.append("  Le soglie delle fasce sono calcolate usando i quantili 33°/66° specifici per ogni tipo di traiettoria.")

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
