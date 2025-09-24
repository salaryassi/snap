#!/usr/bin/env python3
"""
analyse.py

Merged: original per-coin PDF + aggregate analysis
        + altrank backtesting grid and summary export.

Usage examples:
    python analyse.py --db lunarcrush.db --top-n 50 --output-pdf analysis.pdf --run-backtest
    python analyse.py --db lunarcrush.db --symbol BTC --output-pdf btc_analysis.pdf

Notes:
- This script expects an SQLite DB with a 'snapshots' table similar to the earlier format.
- The backtester assumes hourly snapshots; holding periods are measured in hours (1,2,4).
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from datetime import datetime
import numpy as np
import sys
import warnings
from itertools import product

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Utility plotting helpers
# -------------------------
def add_text_page(pdf, text, figsize=(8.5, 11)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.text(0.05, 0.95, text, va='top', ha='left', wrap=True, fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

# -------------------------
# DB / data helpers
# -------------------------
def fetch_symbol_df(conn, symbol):
    """Fetch the snapshot DataFrame for a given symbol (coerced to numeric where appropriate)."""
    query = """
    SELECT snapshot_time, alt_rank, price, galaxy_score, social_volume_24h, interactions_24h, volume_24h, market_cap
    FROM snapshots
    WHERE symbol = ?
    ORDER BY snapshot_time
    """
    df = pd.read_sql_query(query, conn, params=(symbol.upper(),))
    if df.empty:
        return df
    df['snapshot_time'] = pd.to_datetime(df['snapshot_time'], errors='coerce')
    # numeric coercion
    for col in ['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h', 'volume_24h', 'market_cap']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    df = df.dropna(subset=['snapshot_time']).sort_values('snapshot_time').reset_index(drop=True)
    return df

def build_features_for_symbol(df):
    """
    Given a per-symbol dataframe sorted by snapshot_time ascending,
    compute:
      - altrank_improve_1h: A WEIGHTED 3-HOUR score. It rewards sustained AltRank
        improvement and adds a bonus for rising volume.
      - forward returns fut_ret_1h/2h/4h for backtesting.
    """
    if df.empty or len(df) < 4: # Need at least 4 data points for a 3-hour lookback
        return df
    out = df.copy().reset_index(drop=True)

    # --- NEW: Weighted 3-Hour AltRank Improvement Score ---
    
    # Get the AltRank and Volume from the previous 3 hours using shift.
    # shift(1) is t-1 (one hour ago), shift(2) is t-2, etc.
    alt_rank_t0 = out['alt_rank'] # Current hour
    alt_rank_t1 = out['alt_rank'].shift(1) # 1 hour ago
    alt_rank_t2 = out['alt_rank'].shift(2) # 2 hours ago
    alt_rank_t3 = out['alt_rank'].shift(3) # 3 hours ago
    
    volume_t0 = out['volume_24h'] # Current hour
    volume_t1 = out['volume_24h'].shift(1) # 1 hour ago

    # Calculate the improvement (a positive number) for each of the last 3 hours.
    # Improvement = previous_rank - current_rank.
    improve_h1 = alt_rank_t1 - alt_rank_t0 # Most recent hour's improvement
    improve_h2 = alt_rank_t2 - alt_rank_t1 # Improvement from 2 hours ago
    improve_h3 = alt_rank_t3 - alt_rank_t2 # Improvement from 3 hours ago

    # Initialize the score with the most recent hour's improvement.
    # We only care about positive improvements, so set non-improvements to 0.
    score = np.where(improve_h1 > 0, improve_h1, 0)

    # WEIGHTING LOGIC:
    # Add weighted score for sustained improvements.
    
    # If there was an improvement in the last hour AND the hour before,
    # add the previous hour's improvement multiplied by a weight (1.5).
    # This rewards a 2-hour streak.
    score += np.where((improve_h1 > 0) & (improve_h2 > 0), improve_h2 * 1.5, 0)
    
    # If there was an improvement for the last 3 consecutive hours,
    # add the earliest hour's improvement with an even higher weight (2.0).
    # This heavily rewards a sustained 3-hour streak of rank climbing.
    score += np.where((improve_h1 > 0) & (improve_h2 > 0) & (improve_h3 > 0), improve_h3 * 2.0, 0)

    # VOLUME BONUS:
    # Add a fixed bonus if volume is also increasing, confirming momentum.
    volume_is_rising = volume_t0 > volume_t1
    rank_is_improving = improve_h1 > 0
    
    # Where both conditions are true, add a fixed bonus of 5 points to the score.
    score += np.where(volume_is_rising & rank_is_improving, 5, 0)
    
    # Assign the final calculated score to the column used by the backtester.
    # This ensures compatibility with the rest of the script.
    out['altrank_improve_1h'] = score

    # --- Original Feature Engineering (kept for backtesting) ---
    
    # Previous hour's alt rank for reference
    out['alt_rank_prev'] = out['alt_rank'].shift(1)

    # Forward prices (lead) to calculate future returns
    out['price_next1'] = out['price'].shift(-1)
    out['price_next2'] = out['price'].shift(-2)
    out['price_next4'] = out['price'].shift(-4)

    # Forward returns
    for k, col in [(1, 'price_next1'), (2, 'price_next2'), (4, 'price_next4')]:
        out[f'fut_ret_{k}h'] = np.where(out['price'] > 0, (out[col] - out['price']) / out['price'], np.nan)
        
    return out

def build_all_features(conn, symbols=None):
    """Build features for all symbols (or a list). Returns a concatenated dataframe."""
    all_frames = []
    if symbols is None:
        # get distinct symbols
        q = "SELECT DISTINCT symbol FROM snapshots"
        symbols = pd.read_sql_query(q, conn)['symbol'].astype(str).tolist()
    for s in symbols:
        df = fetch_symbol_df(conn, s)
        if df.empty:
            continue
        feats = build_features_for_symbol(df)
        feats['symbol'] = s.upper()
        all_frames.append(feats)
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)

# -------------------------
# Per-coin analysis (kept largely from your original)
# -------------------------
def analyze_coin(conn, symbol, pdf, drop_threshold=10):
    """
    Analyze a single coin, add plots/pages to pdf, and return a summary dict.
    If insufficient data, returns None.
    """
    df = fetch_symbol_df(conn, symbol)
    if df.empty or len(df) < 2:
        print(f"[WARN] Insufficient data for {symbol}")
        return None

    # Detect sudden drops in alt_rank (increase in rank number = worse)
    df['alt_rank_diff'] = df['alt_rank'].diff()
    drops = df[df['alt_rank_diff'] > drop_threshold]

    # Summary calculations
    num_snapshots = len(df)
    num_drops = len(drops)
    avg_alt_diff = float(drops['alt_rank_diff'].mean()) if num_drops > 0 else np.nan

    price_changes = []
    drop_events = []
    if num_drops > 0:
        for drop_idx in drops.index:
            next_idx = drop_idx + 1
            if next_idx < len(df):
                p = df.loc[drop_idx, 'price']
                pn = df.loc[next_idx, 'price']
                if pd.notna(p) and pd.notna(pn) and p != 0:
                    pct = (pn - p) / p * 100
                    price_changes.append(pct)
                    drop_events.append({
                        'symbol': symbol.upper(),
                        'time': df.loc[drop_idx, 'snapshot_time'],
                        'alt_rank_before': df.loc[drop_idx-1, 'alt_rank'] if drop_idx-1 in df.index else np.nan,
                        'alt_rank_after': df.loc[drop_idx, 'alt_rank'],
                        'alt_rank_diff': df.loc[drop_idx, 'alt_rank_diff'],
                        'price_before': p,
                        'price_after': pn,
                        'price_change_pct': pct
                    })
    avg_price_change = float(np.mean(price_changes)) if price_changes else np.nan

    # Correlations
    def safe_corr(a, b):
        if a.isna().all() or b.isna().all():
            return np.nan
        if a.nunique() <= 1 or b.nunique() <= 1:
            return np.nan
        return float(a.corr(b))

    corr_alt_price = safe_corr(df['alt_rank'], df['price'])
    corr_alt_social = safe_corr(df['alt_rank'], df.get('social_volume_24h', pd.Series(dtype=float)))

    summary = {
        'symbol': symbol.upper(),
        'num_snapshots': num_snapshots,
        'num_drops': num_drops,
        'avg_alt_diff_on_drop': avg_alt_diff if not pd.isna(avg_alt_diff) else np.nan,
        'avg_price_change_after_drop': avg_price_change if not pd.isna(avg_price_change) else np.nan,
        'corr_alt_price': corr_alt_price,
        'corr_alt_social_vol': corr_alt_social
    }

    # Explanation text
    explanation = f"""
Analysis for {symbol.upper()}

- Snapshot count: {num_snapshots}
- Detected drops: {num_drops} (threshold: {drop_threshold} alt-rank points)
- Avg alt_rank change on drops: {summary['avg_alt_diff_on_drop']}
- Avg price % change after drop: {summary['avg_price_change_after_drop']}

Notes:
- AltRank: lower is better.
- Drops are flagged when alt_rank increases (worsens) by more than {drop_threshold} between consecutive snapshots.
- Correlations are Pearson r between AltRank and other metrics (NaN if insufficient variation).
"""
    add_text_page(pdf, explanation)

    # Page 1: Time series AltRank & Price
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['snapshot_time'], df['alt_rank'], label='AltRank')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('AltRank')
    ax1.invert_yaxis()  # Lower rank is better
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.plot(df['snapshot_time'], df['price'], label='Price')
    ax2.set_ylabel('Price')
    for d in drops.index:
        t = df.loc[d, 'snapshot_time']
        ax1.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        if d + 1 < len(df):
            p = df.loc[d, 'price']
            pn = df.loc[d + 1, 'price']
            if pd.notna(p) and p != 0 and pd.notna(pn):
                pct = (pn - p) / p * 100
                ax1.annotate(f'+{df.loc[d,"alt_rank_diff"]:.1f}\n{pct:.2f}%', 
                             (t, df.loc[d, 'alt_rank']), 
                             xytext=(10, 10), textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.4', fc='yellow', alpha=0.6),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    fig.suptitle(f'{symbol.upper()} AltRank & Price')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 2: Correlation matrix for this coin (if enough numeric data)
    corr_cols = ['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h']
    corr_df = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45)
    ax.set_yticklabels(corr_df.columns)
    plt.colorbar(im, ax=ax)
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            txtcol = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=txtcol)
    fig.suptitle(f'{symbol.upper()} Correlation Matrix')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 3: Scatter AltRank vs Price
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['alt_rank'], df['price'])
    ax.set_xlabel('AltRank')
    ax.set_ylabel('Price')
    ax.set_title(f'{symbol.upper()} AltRank vs Price')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Page 4: Social Volume vs AltRank timeline
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['snapshot_time'], df['social_volume_24h'], label='Social Volume 24h')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Social Volume 24h')
    ax2 = ax1.twinx()
    ax2.plot(df['snapshot_time'], df['alt_rank'], label='AltRank')
    ax2.invert_yaxis()
    ax2.set_ylabel('AltRank')
    fig.suptitle(f'{symbol.upper()} Social Volume vs AltRank')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    return summary

# -------------------------
# Backtesting utilities
# -------------------------
def backtest_rule(df_all, threshold, hold_hours=2, min_volume=1e6, fee=0.0008, min_trades_filter=0):
    """
    df_all: DataFrame with columns ['symbol','snapshot_time','altrank_improve_1h','fut_ret_1h','fut_ret_2h','fut_ret_4h','volume_24h']
    threshold: minimal altrank_improve_1h to trigger signal (the new weighted score)
    hold_hours: 1,2 or 4
    Returns: dict of aggregate metrics and trades DataFrame
    """
    fut_col = {1: 'fut_ret_1h', 2: 'fut_ret_2h', 4: 'fut_ret_4h'}[hold_hours]
    df = df_all.copy()
    # filter valid rows
    df = df.dropna(subset=['altrank_improve_1h', fut_col, 'volume_24h'])
    signals = df[(df['altrank_improve_1h'] >= threshold) & (df['volume_24h'] >= min_volume)].copy()
    if signals.empty:
        return {
            'trades': 0, 'avg_ret': np.nan, 'win_rate': np.nan, 'sharpe': np.nan, 'cum_return': np.nan,
            'trades_df': signals
        }
    signals['gross_ret'] = signals[fut_col]
    signals['net_ret'] = signals['gross_ret'] - fee
    trades = signals.copy()
    total = len(trades)
    avg = trades['net_ret'].mean()
    wins = (trades['net_ret'] > 0).sum()
    win_rate = wins / total
    std = trades['net_ret'].std(ddof=0) if trades['net_ret'].std(ddof=0) > 0 else 1e-9
    # annualize approximate (assuming hourly samples)
    sharpe = (avg / std) * np.sqrt((365 * 24) / hold_hours) if std > 0 else np.nan
    cum_return = (1 + trades['net_ret']).prod() - 1
    return {
        'trades': total, 'avg_ret': avg, 'win_rate': win_rate, 'sharpe': sharpe, 'cum_return': cum_return,
        'trades_df': trades
    }

def grid_search_backtest(df_all, thresholds=[1,3,5,10,20], holds=[1,2,4], vols=[1e5,1e6,1e7], fee=0.0008):
    rows = []
    # NOTE: With the new weighted score, you might need to test higher thresholds
    # e.g., thresholds=[10, 20, 30, 50]
    for thr, h, v in product(thresholds, holds, vols):
        res = backtest_rule(df_all, threshold=thr, hold_hours=h, min_volume=v, fee=fee)
        rows.append({
            'threshold': thr, 'hold_hours': h, 'min_volume': v,
            'trades': res['trades'], 'avg_ret': res['avg_ret'],
            'win_rate': res['win_rate'], 'sharpe': res['sharpe'], 'cum_return': res['cum_return']
        })
    return pd.DataFrame(rows)

def permutation_pvalue(arr, observed_mean, n=2000):
    """Permutation test for mean >= observed_mean"""
    if len(arr) == 0:
        return 1.0
    vals = arr.copy()
    greater = 0
    for _ in range(n):
        np.random.shuffle(vals)
        if vals.mean() >= observed_mean:
            greater += 1
    return (greater + 1) / (n + 1)

# -------------------------
# Aggregate plotting additions for backtest results
# -------------------------
def add_backtest_pages(pdf, df_all, grid_df, best_conf, trades_df):
    add_text_page(pdf, f"Backtest summary\nGenerated: {datetime.utcnow().isoformat()}Z\n\nBest config: {best_conf.to_dict() if best_conf is not None else 'N/A'}")

    # Grid heatmap of average returns (pivot)
    try:
        heat = grid_df.pivot_table(index='min_volume', columns='hold_hours', values='avg_ret', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(heat.fillna(0).values, aspect='auto', cmap='RdYlGn')
        ax.set_yticks(range(len(heat.index))); ax.set_yticklabels([int(x) for x in heat.index])
        ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels([int(x) for x in heat.columns])
        ax.set_xlabel('Hold Hours'); ax.set_ylabel('Min Volume')
        ax.set_title('Grid: average net return (threshold varying rows)')
        plt.colorbar(im, ax=ax)
        pdf.savefig(fig); plt.close(fig)
    except Exception:
        pass

    # Histogram of trade returns for best config
    if trades_df is not None and not trades_df.empty:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(trades_df['net_ret'].dropna(), bins=40)
        ax.set_title('Distribution of trade net returns (best config)')
        ax.set_xlabel('Net return per trade')
        ax.set_ylabel('Frequency')
        pdf.savefig(fig); plt.close(fig)

        # Top coins by average net return
        per_coin = trades_df.groupby('symbol')['net_ret'].agg(['count','mean']).reset_index().sort_values('mean', ascending=False)
        top = per_coin.head(30)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(top['symbol'], top['mean'])
        ax.set_xticklabels(top['symbol'], rotation=45, ha='right')
        ax.set_title('Top coins by avg trade net return (best config)')
        ax.set_ylabel('Avg net return')
        plt.tight_layout()
        pdf.savefig(fig); plt.close(fig)

# -------------------------
# Aggregate plots (your original)
# -------------------------
def aggregate_plots(summaries_df, pdf):
    """Create aggregate plots (across all coins) and append them to the PDF."""
    # Ensure numeric
    for col in ['corr_alt_price', 'corr_alt_social_vol', 'avg_price_change_after_drop', 'num_drops']:
        if col in summaries_df.columns:
            summaries_df[col] = pd.to_numeric(summaries_df[col], errors='coerce')

    add_text_page(pdf, "Aggregate analysis across all coins\n\nBelow are distributional views and cross-coin comparisons.")

    # Histogram of AltRank vs Price correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    data = summaries_df['corr_alt_price'].dropna()
    if len(data) > 0:
        ax.hist(data, bins=20)
    ax.set_title("Distribution of AltRank ? Price Correlation (across coins)")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Frequency")
    pdf.savefig(fig)
    plt.close(fig)

    # Histogram of AltRank vs Social Volume correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    data2 = summaries_df['corr_alt_social_vol'].dropna()
    if len(data2) > 0:
        ax.hist(data2, bins=20)
    ax.set_title("Distribution of AltRank ? Social Volume Correlation (across coins)")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Frequency")
    pdf.savefig(fig)
    plt.close(fig)

    # Scatter: corr_alt_price vs corr_alt_social_vol (annotated)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(summaries_df['corr_alt_price'], summaries_df['corr_alt_social_vol'])
    for _, row in summaries_df.dropna(subset=['corr_alt_price', 'corr_alt_social_vol']).iterrows():
        ax.annotate(row['symbol'], (row['corr_alt_price'], row['corr_alt_social_vol']), fontsize=8, alpha=0.8)
    ax.set_xlabel("AltRank ? Price Correlation")
    ax.set_ylabel("AltRank ? Social Vol Correlation")
    ax.set_title("Coin-by-coin: correlation comparison")
    pdf.savefig(fig)
    plt.close(fig)

    # Bar: top coins by absolute correlation with price (top 20)
    if 'corr_alt_price' in summaries_df.columns:
        dfc = summaries_df.dropna(subset=['corr_alt_price']).assign(abs_corr=lambda d: d['corr_alt_price'].abs())
        dfc = dfc.sort_values('abs_corr', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(dfc['symbol'], dfc['corr_alt_price'])
        ax.set_xticklabels(dfc['symbol'], rotation=45, ha='right')
        ax.set_title("Top 20 coins by |AltRank ? Price correlation|")
        ax.set_ylabel("Pearson r (signed)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Boxplot: avg_price_change_after_drop distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    clean = summaries_df['avg_price_change_after_drop'].dropna()
    if len(clean) > 0:
        ax.boxplot(clean)
    ax.set_title("Distribution of avg % price change after detected drops (per coin)")
    ax.set_ylabel("Avg % price change")
    pdf.savefig(fig)
    plt.close(fig)

    # Heatmap of summary-level correlations (cross-coin)
    numeric_cols = ['num_snapshots', 'num_drops', 'avg_alt_diff_on_drop', 'avg_price_change_after_drop', 'corr_alt_price', 'corr_alt_social_vol']
    existing = [c for c in numeric_cols if c in summaries_df.columns]
    if len(existing) >= 2:
        corrmat = summaries_df[existing].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corrmat, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corrmat.columns)))
        ax.set_yticks(np.arange(len(corrmat.columns)))
        ax.set_xticklabels(corrmat.columns, rotation=45, ha='right')
        ax.set_yticklabels(corrmat.columns)
        plt.colorbar(im, ax=ax)
        for i in range(len(corrmat.columns)):
            for j in range(len(corrmat.columns)):
                val = corrmat.iloc[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color='white' if abs(val) > 0.5 else 'black')
        ax.set_title("Cross-coin summary correlation matrix")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    add_text_page(pdf, "End of aggregate analysis.")

# -------------------------
# CLI and main
# -------------------------
def get_top_symbols(db_path, top_n):
    """Return top-N symbols by market_cap at the most recent snapshot_time."""
    conn = sqlite3.connect(db_path)
    latest_time_df = pd.read_sql("SELECT MAX(snapshot_time) as max_time FROM snapshots", conn)
    latest_time = latest_time_df['max_time'].iloc[0]
    if latest_time is None:
        conn.close()
        return []
    query = """
    SELECT symbol 
    FROM snapshots
    WHERE snapshot_time = ?
    ORDER BY market_cap DESC
    LIMIT ?
    """
    top_df = pd.read_sql_query(query, conn, params=(latest_time, top_n))
    conn.close()
    return top_df['symbol'].astype(str).tolist()

def main():
    parser = argparse.ArgumentParser(description="Analyze LunarCrush snapshots DB and create combined PDF with aggregate charts and optional altrank backtest.")
    parser.add_argument("--db", default="lunarcrush.db", help="SQLite DB path")
    parser.add_argument("--symbol", help="Single coin symbol to analyze")
    parser.add_argument("--top-n", type=int, help="Analyze top N coins by latest market cap")
    parser.add_argument("--output-pdf", default="analysis.pdf", help="Output PDF file")
    parser.add_argument("--summary-csv", default="summary.csv", help="Output summary CSV file")
    parser.add_argument("--drop-threshold", type=int, default=10, help="Threshold for AltRank drop detection")
    parser.add_argument("--run-backtest", action='store_true', help="Run altrank backtest grid across selected coins and add results to PDF")
    parser.add_argument("--min-volume-default", type=float, default=1e6, help="Default min volume used for reporting / filtering")
    # You might want to adjust the default thresholds for the new, higher-value weighted score
    parser.add_argument("--thresholds", nargs='*', type=float, default=[10, 20, 30, 50, 75], help="Thresholds for the new weighted altrank improvement score")
    parser.add_argument("--holds", nargs='*', type=int, default=[1,2,4], help="Holding periods (hours) to test")
    parser.add_argument("--vols", nargs='*', type=float, default=[1e5,1e6,1e7], help="Volume filters to test in grid")
    parser.add_argument("--fee", type=float, default=0.0008, help="Per-trade fee/slippage to subtract from returns (decimal)")
    args = parser.parse_args()

    if not args.symbol and not args.top_n:
        sys.exit("Must provide either --symbol or --top-n")

    conn = sqlite3.connect(args.db)
    symbols = []
    if args.top_n:
        symbols = get_top_symbols(args.db, args.top_n)
        if not symbols:
            conn.close()
            sys.exit("No data available for top coins")
        print(f"Analyzing top {len(symbols)} coins: {', '.join(symbols)}")
    elif args.symbol:
        symbols = [args.symbol.upper()]

    summaries = []
    # Build features for all requested symbols (used for backtest)
    all_features = build_all_features(conn, symbols=symbols)
    # Save the feature table locally (helpful)
    if not all_features.empty:
        all_features.to_csv('features_all_symbols.csv', index=False)

    # Create PDF and generate per-coin pages
    with PdfPages(args.output_pdf) as pdf:
        for sym in symbols:
            summary = analyze_coin(conn, sym, pdf, args.drop_threshold)
            if summary:
                summaries.append(summary)

        # After per-coin pages, add aggregate pages if we have more than one coin
        if summaries:
            df_summary = pd.DataFrame(summaries)
            add_text_page(pdf, f"Aggregate summary for {len(df_summary)} coins\nGenerated: {datetime.utcnow().isoformat()}Z")
            aggregate_plots(df_summary, pdf)
        else:
            print("[WARN] No summaries collected; skipping aggregate plots.")

        # Run backtest grid (optional)
        if args.run_backtest:
            if all_features.empty:
                add_text_page(pdf, "Backtest requested but no features were available.")
            else:
                # Use the new weighted score in altrank_improve_1h
                df_all = all_features.copy()
                # Keep only useful columns
                needed_cols = ['symbol','snapshot_time','altrank_improve_1h','fut_ret_1h','fut_ret_2h','fut_ret_4h','volume_24h']
                for c in needed_cols:
                    if c not in df_all.columns:
                        df_all[c] = np.nan
                df_back = df_all[needed_cols].dropna(subset=['snapshot_time']).copy()
                # Run grid
                add_text_page(pdf, "Running altrank backtest grid with new 3-hour weighted score. This searches thresholds, holding periods and volume filters.")
                grid_df = grid_search_backtest(df_back, thresholds=args.thresholds, holds=args.holds, vols=args.vols, fee=args.fee)
                grid_df.to_csv('backtest_grid_results.csv', index=False)

                # Choose best by sharpe (or avg_ret if nan sharpe)
                recomputed_rows = []
                for _, row in grid_df.iterrows():
                    thr = row['threshold']; h = int(row['hold_hours']); v = row['min_volume']
                    r = backtest_rule(df_back, threshold=thr, hold_hours=h, min_volume=v, fee=args.fee)
                    recomputed_rows.append({**row, 'trades': r['trades'], 'avg_ret': r['avg_ret'], 'win_rate': r['win_rate'], 'sharpe': r['sharpe'], 'cum_return': r['cum_return']})
                grid_df2 = pd.DataFrame(recomputed_rows).sort_values('sharpe', ascending=False)
                grid_df2.to_csv('backtest_grid_results_recomputed.csv', index=False)

                # pick best config with at least some trades
                best_row = grid_df2[grid_df2['trades'] > 0].sort_values(['sharpe','avg_ret'], ascending=False).head(1)
                best_conf = best_row.iloc[0] if not best_row.empty else None

                # compute trades_df for best conf
                best_trades_df = None
                if best_conf is not None:
                    best_conf_dict = best_conf.to_dict()
                    best_thr = best_conf_dict['threshold']
                    best_hold = int(best_conf_dict['hold_hours'])
                    best_vol = best_conf_dict['min_volume']
                    best_res = backtest_rule(df_back, threshold=best_thr, hold_hours=best_hold, min_volume=best_vol, fee=args.fee)
                    best_trades_df = best_res['trades_df']
                    # save per-trade file
                    if best_trades_df is not None and not best_trades_df.empty:
                        best_trades_df.to_csv('best_conf_trades.csv', index=False)

                    # permutation p-value on net returns
                    obs_mean = best_trades_df['net_ret'].mean() if best_trades_df is not None and not best_trades_df.empty else 0.0
                    pval = permutation_pvalue(best_trades_df['net_ret'].values.copy(), obs_mean, n=2000) if best_trades_df is not None and not best_trades_df.empty else 1.0
                else:
                    pval = 1.0

                # Add backtest pages
                add_backtest_pages(pdf, df_back, grid_df2, best_conf, best_trades_df)

                # Export per-coin trade stats for the best conf (if present)
                if best_trades_df is not None and not best_trades_df.empty:
                    per_coin_stats = best_trades_df.groupby('symbol').agg(trades=('net_ret','count'), avg_net_ret=('net_ret','mean'), win_rate=('net_ret', lambda x: (x>0).mean())).reset_index()
                    per_coin_stats.to_csv('per_coin_trade_stats.csv', index=False)
                    add_text_page(pdf, f"Backtest best config p-value (permutation test): {pval:.4f}\nSee per_coin_trade_stats.csv and best_conf_trades.csv for detail.")
                else:
                    add_text_page(pdf, "No trades found for best config (or best config absent).")

    conn.close()

    # write summaries CSV
    if summaries:
        pd.DataFrame(summaries).to_csv(args.summary_csv, index=False)
        print(f"[OK] Summary exported to {args.summary_csv}")
    # grid results saved as CSV earlier
    print(f"[OK] Analysis exported to {args.output_pdf}")
    print("Additional files (if generated): features_all_symbols.csv, backtest_grid_results.csv, backtest_grid_results_recomputed.csv, best_conf_trades.csv, per_coin_trade_stats.csv")

if __name__ == "__main__":
    main()