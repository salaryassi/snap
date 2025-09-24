#!/usr/bin/env python3
"""
analyze_pipeline_outputs.py

Scan directory for pipeline output files (model portfolio CSVs and per-symbol stats),
compute metrics per horizon, generate a PDF report with visualizations and plain-English
explanations about what the numbers mean.

Usage:
    python analyze_pipeline_outputs.py --out pdf_name(optional)

Outputs:
    - pipeline_analysis_report.pdf (default)
    - pipeline_analysis_summary.csv
"""

import argparse
import glob
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import random
from datetime import datetime

# -------------------------
# Helpers
# -------------------------
def find_files(pattern):
    return sorted(glob.glob(pattern))

def infer_horizon_from_filename(fname):
    """
    Look for pattern like _2h, _6h, _24h in filename and return '2', '6', '24'
    Otherwise return None.
    """
    m = re.search(r'_(\d+)h', fname)
    if m:
        return int(m.group(1))
    return None

def load_portfolio_csvs():
    """
    Find all files matching '*_model_portfolio.csv' and return mapping horizon -> filepath.
    """
    files = find_files("*_model_portfolio.csv")
    out = {}
    for f in files:
        h = infer_horizon_from_filename(f)
        if h is not None:
            out.setdefault(h, {})['portfolio'] = f
    # also find predictions & per_symbol_stats (optional)
    for f in find_files("*_model_predictions.csv"):
        h = infer_horizon_from_filename(f)
        if h is not None:
            out.setdefault(h, {})['predictions'] = f
    for f in find_files("*_per_symbol_stats.csv"):
        h = infer_horizon_from_filename(f)
        if h is not None:
            out.setdefault(h, {})['per_symbol'] = f
    # also rule portfolio
    for f in find_files("*_rule_portfolio.csv"):
        h = infer_horizon_from_filename(f)
        if h is not None:
            out.setdefault(h, {})['rule'] = f
    return out

def read_portfolio(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, parse_dates=['snapshot_time'], infer_datetime_format=True)
    except Exception:
        df = pd.read_csv(path)
    # normalize column names
    if 'return' not in df.columns and 'returns' in df.columns:
        df.rename(columns={'returns':'return'}, inplace=True)
    return df

def compute_summary_from_portfolio(portf_df):
    """
    portf_df: DataFrame with columns ['snapshot_time','return', ...]
    returns dictionary of summary stats
    """
    s = {}
    if portf_df is None or portf_df.empty:
        s.update({'count':0,'mean':np.nan,'std':np.nan,'median':np.nan,'min':np.nan,'max':np.nan,
                  'final_growth':np.nan,'win_frac':np.nan})
        return s
    arr = portf_df['return'].dropna().astype(float)
    s['count'] = len(arr)
    s['mean'] = float(arr.mean())
    s['std'] = float(arr.std(ddof=1)) if len(arr)>1 else float(0.0)
    s['median'] = float(arr.median())
    s['min'] = float(arr.min())
    s['max'] = float(arr.max())
    # final cumulative growth
    cum = (1 + arr).cumprod()
    s['final_growth'] = float(cum.iloc[-1]) if len(cum)>0 else float(np.nan)
    s['win_frac'] = float((arr > 0).mean()) if len(arr)>0 else float(np.nan)
    return s

def permutation_pvalue(arr, n_iter=2000, seed=42):
    """
    Empirical p-value for mean(arr) > 0 using permutation of signs or shuffling.
    We'll use shuffling to preserve distribution but remove time structure.
    """
    arr = np.asarray(arr.dropna().astype(float))
    if len(arr) == 0:
        return np.nan
    np.random.seed(seed)
    obs = float(np.mean(arr))
    count = 0
    for _ in range(n_iter):
        perm = np.random.permutation(arr)
        if perm.mean() >= obs:
            count += 1
    pval = (count + 1) / (n_iter + 1)
    return pval

def make_text_figure(text_lines, title=None, figsize=(8.27, 11.69)):
    """
    Create a matplotlib figure with wrapped text lines. Return fig.
    Default figsize is A4 in inches.
    """
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    if title:
        plt.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, weight='bold')
    y = 0.88
    for line in text_lines:
        plt.text(0.02, y, line, ha='left', va='top', fontsize=9, family='monospace')
        y -= 0.03
        if y < 0.05:
            break
    return fig

def table_figure(df_table, title=None, figsize=(8.27, 11.69)):
    """
    Render a small DataFrame as a table on a matplotlib page.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    if title:
        plt.title(title)
    # convert to string for safe display
    str_df = df_table.round(6).astype(object)
    # draw table
    table = ax.table(cellText=str_df.values,
                     colLabels=str_df.columns,
                     rowLabels=str_df.index,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    return fig

# -------------------------
# Explanations generator
# -------------------------
def interpret_summary(h, s, pval):
    """
    Build plain English interpretation based on summary stats.
    """
    lines = []
    lines.append(f"Horizon: {h} hours")
    if s['count'] == 0:
        lines.append("No portfolio periods were found (no predictions/backtest).")
        return lines
    lines.append(f"Number of test periods (portfolio dates): {s['count']}")
    lines.append(f"Average portfolio return per period: {s['mean']*100:.3f}%")
    lines.append(f"Median period return: {s['median']*100:.3f}%")
    lines.append(f"Std. dev. of period returns: {s['std']*100:.3f}%")
    lines.append(f"Win fraction (periods with positive return): {s['win_frac']*100:.1f}%")
    lines.append(f"Final cumulative growth (equity multiplier): {s['final_growth']:.4f}")
    if s['final_growth'] > 1.0:
        lines.append("=> Overall the strategy *grew* capital across the test periods.")
    elif s['final_growth'] < 1.0:
        lines.append("=> Overall the strategy *lost* capital across the test periods.")
    else:
        lines.append("=> No net change in capital across the test periods.")
    lines.append("")  # blank

    # Statistical comment via permutation p-value
    if np.isnan(pval):
        lines.append("Could not compute statistical significance (no returns).")
    else:
        lines.append(f"Permutation test p-value for mean(return) > 0: {pval:.3f}")
        if pval < 0.05:
            lines.append("=> The positive mean return is unlikely under the null (p < 0.05).")
        else:
            lines.append("=> No strong evidence the mean return is different from zero.")
    lines.append("")

    # Practical considerations
    lines.append("Practical notes:")
    lines.append("- These are *period* returns averaged across timestamps where the model made picks.")
    lines.append("- Transaction costs and slippage were included when the pipeline computed net returns. If you want to see gross edge, re-run pipeline with transaction_cost=0.")
    lines.append("- A high win fraction with low mean return often indicates many small wins and few large losses (or vice versa). Check per-symbol stats to see which tokens drive returns.")
    lines.append("- If final growth is < 1 but win fraction > 50%, the losses are larger when they happen; consider increasing threshold or targeting fewer, larger moves.")
    return lines

# -------------------------
# Main report builder
# -------------------------
def build_report(output_pdf="pipeline_analysis_report.pdf"):
    filemap = load_portfolio_csvs()
    if not filemap:
        print("No model_portfolio outputs found in current directory (pattern '*_model_portfolio.csv').")
        return

    # sort horizons ascending
    horizons = sorted(filemap.keys())
    horizon_summaries = {}
    per_horizon_details = {}

    # PDF
    with PdfPages(output_pdf) as pdf:
        # Cover page
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        cover_lines = [
            "Pipeline Outputs Analysis Report",
            f"Generated: {now}",
            "",
            "This report summarizes model backtest outputs produced by the pipeline.",
            "It computes numeric summaries, runs a simple permutation test,",
            "and provides a plain-English interpretation per horizon.",
            "",
            "Files scanned:",
        ]
        for h in horizons:
            d = filemap[h]
            flist = [d.get(k, "<missing>") for k in ('portfolio','predictions','per_symbol','rule')]
            cover_lines.append(f"  - {h}h: portfolio={flist[0]}, predictions={flist[1]}, per_symbol={flist[2]}, rule={flist[3]}")
        fig = make_text_figure(cover_lines, title="Pipeline Outputs Analysis Report")
        pdf.savefig(fig); plt.close(fig)

        # Per-horizon pages
        for h in horizons:
            d = filemap[h]
            port_path = d.get('portfolio')
            port_df = read_portfolio_csvsafely = None
            try:
                port_df = read_portfolio(port_path) if port_path else None
            except Exception as ex:
                print(f"Error reading portfolio for {h}h: {ex}")
                port_df = None

            summary = compute_summary_from_portfolio(port_df)
            # permutation pvalue
            pval = permutation_pvalue(port_df['return']) if (port_df is not None and 'return' in port_df.columns) else np.nan
            horizon_summaries[h] = (summary, pval)
            per_horizon_details[h] = {'portfolio': port_df}

            # Explanation page
            interp_lines = interpret_summary(h, summary, pval)
            fig = make_text_figure(interp_lines, title=f"Horizon {h}h Summary")
            pdf.savefig(fig); plt.close(fig)

            # Cumulative growth plot
            if port_df is not None and not port_df.empty:
                fig, ax = plt.subplots(figsize=(10,4.5))
                arr = port_df['return'].dropna().astype(float)
                cum = (1 + arr).cumprod()
                ax.plot(cum.index, cum.values)
                ax.set_title(f"Cumulative growth — horizon {h}h (top-N picks)")
                ax.set_xlabel("Index (period)")
                ax.set_ylabel("Growth")
                pdf.savefig(fig); plt.close(fig)

                # distribution plot
                fig, ax = plt.subplots(figsize=(10,4.5))
                ax.hist(arr, bins=80)
                ax.set_title(f"Distribution of period returns — horizon {h}h")
                ax.set_xlabel("Return")
                ax.set_ylabel("Frequency")
                pdf.savefig(fig); plt.close(fig)

            # Per-symbol top table (if exists)
            per_sym_path = d.get('per_symbol')
            if per_sym_path and os.path.exists(per_sym_path):
                try:
                    per_sym = pd.read_csv(per_sym_path, index_col=0)
                    # keep top 10 by mean
                    top10 = per_sym.sort_values('mean', ascending=False).head(10)
                    # put table page
                    fig = table_figure(top10, title=f"Top 10 symbols by mean realized return — {h}h")
                    pdf.savefig(fig); plt.close(fig)
                except Exception as ex:
                    print(f"Error reading per_symbol for {h}h: {ex}")

        # Aggregated summary page (table)
        rows = []
        for h, (s, p) in horizon_summaries.items():
            rows.append({
                'horizon_h': h,
                'periods': s['count'],
                'mean_return': s['mean'],
                'median': s['median'],
                'std': s['std'],
                'final_growth': s['final_growth'],
                'win_frac': s['win_frac'],
                'perm_pvalue': p
            })
        summary_df = pd.DataFrame(rows).set_index('horizon_h').sort_index()
        # write CSV summary
        summary_df.to_csv("pipeline_analysis_summary.csv")
        # add to pdf
        fig = table_figure(summary_df, title="Aggregate summary across horizons")
        pdf.savefig(fig); plt.close(fig)

        # final recommendations page
        recomms = [
            "Recommendations (automatically generated):",
            "",
            "- If final_growth <= 1.0 for all horizons: the strategy lost capital in the backtest. Consider:",
            "  * raising pos_threshold (target larger moves),",
            "  * increasing entry/exit horizon if you suspect alt_rank signals play out slower,",
            "  * restricting to top-volume symbols (increase --min-mean-volume),",
            "  * adding / experimenting with social features or regressions for ranking.",
            "",
            "- If final_growth > 1.0 for some horizon but not others: focus on the winning horizon(s) and",
            "  test more robustly (out-of-sample time windows, slippage modeling, position sizing).",
            "",
            "- Inspect per-symbol CSVs (per-horizon *_per_symbol_stats.csv). If a few symbols drive returns,",
            "  consider a per-symbol strategy or train specialized models for those symbols.",
            "",
            "- Always verify gross edge (run pipeline with transaction_cost=0) to see if costs remove the edge.",
            "",
            "End of automated report."
        ]
        fig = make_text_figure(recomms, title="Automated Recommendations")
        pdf.savefig(fig); plt.close(fig)

    print(f"Report written to: {output_pdf}")
    print("Numeric summary written to: pipeline_analysis_summary.csv")
    return summary_df

def load_portfolio_csvs():
    """
    Find all files with pattern '*_model_portfolio.csv' and return mapping horizon -> dict(paths)
    """
    files = find_files("*_model_portfolio.csv")
    out = {}
    for f in files:
        h = infer_horizon_from_filename(f)
        if h is None:
            continue
        out.setdefault(h, {})['portfolio'] = f
    # predictions
    for f in find_files("*_model_predictions.csv"):
        h = infer_horizon_from_filename(f)
        if h is None:
            continue
        out.setdefault(h, {})['predictions'] = f
    # per-symbol
    for f in find_files("*_per_symbol_stats.csv"):
        h = infer_horizon_from_filename(f)
        if h is None:
            continue
        out.setdefault(h, {})['per_symbol'] = f
    # rule
    for f in find_files("*_rule_portfolio.csv"):
        h = infer_horizon_from_filename(f)
        if h is None:
            continue
        out.setdefault(h, {})['rule'] = f
    return out

# small wrapper to avoid name clash above
def load_portfolio_csvs_wrapper():
    return load_portfolio_csvs()

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='pipeline_analysis_report.pdf', help='Output PDF filename')
    args = parser.parse_args()
    # run the report builder
    build_report(output_pdf=args.out)
