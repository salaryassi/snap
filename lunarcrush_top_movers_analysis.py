#!/usr/bin/env python3
"""
movers_with_backtest.py

Select top-N coins by market cap, choose top-K movers by first-hours AltRank movement,
then run chronological (first-hit) TP/SL backtests across selected coins and produce
aggregate analysis + PDF report + CSV outputs.

Usage:
    python movers_with_backtest.py --db lunarcrush.db --top-n 50 --select-k 20 --hours 3 --freq H --output-pdf movers_report.pdf
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import sys
from datetime import timedelta, datetime

# -------------------- Utilities --------------------
def get_top_symbols(db_path, top_n):
    conn = sqlite3.connect(db_path)
    try:
        latest_time_df = pd.read_sql("SELECT MAX(snapshot_time) as max_time FROM snapshots", conn)
        latest_time = latest_time_df['max_time'].iloc[0]
        if latest_time is None:
            return []
        query = """
        SELECT symbol
        FROM snapshots
        WHERE snapshot_time = ?
        ORDER BY market_cap DESC
        LIMIT ?
        """
        top_df = pd.read_sql_query(query, conn, params=(latest_time, top_n))
        return top_df['symbol'].astype(str).tolist()
    finally:
        conn.close()

def fetch_snapshots_for_symbols(db_path, symbols, cols=None):
    if not symbols:
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        if cols is None:
            cols = ['snapshot_time', 'symbol', 'alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h']
        col_sql = ", ".join(cols)
        placeholders = ",".join("?" for _ in symbols)
        query = f"""
        SELECT {col_sql}
        FROM snapshots
        WHERE symbol IN ({placeholders})
        ORDER BY snapshot_time
        """
        df = pd.read_sql_query(query, conn, params=tuple([s.upper() for s in symbols]))
    finally:
        conn.close()
    if df.empty:
        return df
    df['snapshot_time'] = pd.to_datetime(df['snapshot_time'], errors='coerce')
    numeric_cols = [c for c in cols if c not in ('snapshot_time', 'symbol')]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    df = df.dropna(subset=['snapshot_time']).reset_index(drop=True)
    return df

def compute_first_n_hours_movement(df, hours=3):
    rows = []
    grouped = df.groupby('symbol')
    for symbol, g in grouped:
        g = g.sort_values('snapshot_time').reset_index(drop=True)
        if 'alt_rank' not in g.columns or g['alt_rank'].dropna().empty:
            continue
        t0 = g['snapshot_time'].iloc[0]
        t_end = t0 + timedelta(hours=hours)
        window = g[(g['snapshot_time'] >= t0) & (g['snapshot_time'] <= t_end)].copy()
        if window.empty:
            continue
        first_alt = float(window['alt_rank'].iloc[0]) if pd.notna(window['alt_rank'].iloc[0]) else np.nan
        last_alt = float(window['alt_rank'].iloc[-1]) if pd.notna(window['alt_rank'].iloc[-1]) else np.nan
        if np.isnan(first_alt) or np.isnan(last_alt):
            continue
        movement = abs(last_alt - first_alt)
        rows.append({
            'symbol': symbol,
            't0': t0,
            't_end': t_end,
            'first_alt_rank': first_alt,
            'last_alt_rank': last_alt,
            'abs_movement': movement,
            'samples_in_window': len(window)
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('abs_movement', ascending=False).reset_index(drop=True)

def time_bucket_and_average(df, freq="H", metrics=None):
    if df.empty:
        return pd.DataFrame(), df
    if metrics is None:
        metrics = ['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h']
    df = df.copy()
    df['time_bucket'] = df['snapshot_time'].dt.floor(freq)
    df_bucket_mean = df.groupby('time_bucket')[metrics].mean().reset_index().sort_values('time_bucket')
    return df_bucket_mean, df

def safe_corr_matrix(df, cols):
    dfc = df[cols].copy()
    dfc = dfc.dropna(axis=1, how='all')
    nunique = dfc.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        dfc = dfc.drop(columns=constant_cols)
    if dfc.shape[1] < 2:
        return pd.DataFrame()
    return dfc.corr()

# -------------------- Backtest (first-hit chronological) --------------------
def run_first_hit_backtest(df, symbol_col='symbol', time_col='snapshot_time', price_col='price',
                           strategies=None, initial_investment=10.0):
    """
    df: coin-snapshot rows (must contain symbol, time_col, price_col)
    strategies: dict name -> (tp_pct, sl_pct)
    Returns:
      - strategy_summary_df: per-strategy totals
      - per_symbol_outcomes_df: rows with symbol, strategy, outcome, hit_time, hit_price, pnl, notes
    """
    if strategies is None:
        strategies = {
            "TP 2% / SL 1%": (0.02, 0.01),
            "TP 4% / SL 2%": (0.04, 0.02),
            "TP 5% / SL 2.5%": (0.05, 0.025),
            "TP 6% / SL 2%": (0.06, 0.02),
            "TP 6% / SL 3%": (0.06, 0.03),
            "TP 9% / SL 3%": (0.09, 0.03),
            "TP 10% / SL 5%": (0.10, 0.05),
            "TP 12% / SL 6%": (0.12, 0.06),
            "TP 15% / SL 7.5%": (0.15, 0.075)
        }

    symbols = sorted(df[symbol_col].unique())
    per_symbol_rows = []

    # Pre-sort
    df_sorted = df.sort_values([symbol_col, time_col]).reset_index(drop=True)

    for name, (tp_pct, sl_pct) in strategies.items():
        total_pnl = 0.0
        wins = 0
        losses = 0
        unevaluated = 0

        for sym in symbols:
            sym_rows = df_sorted[df_sorted[symbol_col] == sym].reset_index(drop=True)
            if sym_rows.empty:
                continue
            # entry price: first available price for that symbol
            entry_price = float(sym_rows[price_col].iloc[0])
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)

            outcome = 'no_hit'  # default if neither TP nor SL hit
            hit_time = None
            hit_price = None
            # iterate rows chronologically and stop at first hit
            for _, r in sym_rows.iterrows():
                p = float(r[price_col])
                if p >= tp_price:
                    outcome = 'tp'
                    hit_time = r[time_col]
                    hit_price = p
                    break
                if p <= sl_price:
                    outcome = 'sl'
                    hit_time = r[time_col]
                    hit_price = p
                    break

            if outcome == 'tp':
                pnl = initial_investment * tp_pct
                total_pnl += pnl
                wins += 1
            elif outcome == 'sl':
                pnl = - initial_investment * sl_pct
                total_pnl += pnl
                losses += 1
            else:
                # no explicit hit; treat as unrealized (optionally mark final price pnl)
                final_price = float(sym_rows[price_col].iloc[-1])
                # optional: final unrealized pnl percentage
                pnl_pct = (final_price - entry_price) / entry_price
                pnl = initial_investment * pnl_pct
                # decide here whether to include or exclude unrealized; we will record but not count as win/loss
                unevaluated += 1

            per_symbol_rows.append({
                'strategy': name,
                'symbol': sym,
                'entry_time': sym_rows[time_col].iloc[0],
                'entry_price': entry_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'outcome': outcome,
                'hit_time': hit_time,
                'hit_price': hit_price,
                'realized_pnl_usd': pnl if outcome in ('tp','sl') else (pnl if outcome == 'no_hit' else np.nan),
                'final_price': float(sym_rows[price_col].iloc[-1]),
                'final_unrealized_pnl_usd': pnl if outcome == 'no_hit' else (0.0 if outcome in ('tp','sl') else np.nan)
            })

        trades_evaluated = wins + losses
        win_rate = (wins / trades_evaluated * 100) if trades_evaluated > 0 else 0.0
        strategy_summary = {
            'strategy': name,
            'total_pnl_usd': total_pnl,
            'wins': wins,
            'losses': losses,
            'trades_evaluated': trades_evaluated,
            'unevaluated_or_no_hit': unevaluated,
            'win_rate_pct': win_rate
        }
        # accumulate summary rows (we'll build DF later)
        per_symbol_rows.append({'_summary_marker_for_strategy': True, **strategy_summary})

    # Now build summary DF and per-symbol DF by splitting the per_symbol_rows
    summary_rows = []
    symbol_rows = []
    for r in per_symbol_rows:
        if r.get('_summary_marker_for_strategy'):
            # copy summary keys
            summary_rows.append({
                'strategy': r['strategy'],
                'total_pnl_usd': r['total_pnl_usd'],
                'wins': r['wins'],
                'losses': r['losses'],
                'trades_evaluated': r['trades_evaluated'],
                'unevaluated_or_no_hit': r['unevaluated_or_no_hit'],
                'win_rate_pct': r['win_rate_pct']
            })
        else:
            symbol_rows.append(r)

    strategy_summary_df = pd.DataFrame(summary_rows).set_index('strategy').sort_values('total_pnl_usd', ascending=False)
    per_symbol_outcomes_df = pd.DataFrame(symbol_rows)
    return strategy_summary_df, per_symbol_outcomes_df

# -------------------- Main flow --------------------
def main():
    parser = argparse.ArgumentParser(description="Top-movers + backtest analysis")
    parser.add_argument("--db", required=True, help="SQLite DB path")
    parser.add_argument("--top-n", type=int, default=50, help="Get top-N coins by market_cap to consider (default 50)")
    parser.add_argument("--select-k", type=int, default=50, help="From top-N, select top-K movers by 1st-hours AltRank movement")
    parser.add_argument("--hours", type=float, default=3.0, help="Hours window from first snapshot to measure movement (default 3)")
    parser.add_argument("--freq", default="H", help="Time-bucket frequency for averaging (pandas offset alias, default 'H')")
    parser.add_argument("--metrics", nargs="+", default=['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h'],
                        help="Metrics to include in aggregation")
    parser.add_argument("--output-pdf", default="movers_analysis.pdf", help="Output PDF report")
    parser.add_argument("--output-prefix", default="movers_analysis", help="Prefix for CSV outputs")
    parser.add_argument("--initial-investment", type=float, default=10.0, help="Per-trade capital for PnL calculations")
    args = parser.parse_args()

    # 1) get top-N by market cap
    top_symbols = get_top_symbols(args.db, args.top_n)
    if not top_symbols:
        print("[FATAL] No top symbols found. Is DB path/table correct and populated?")
        sys.exit(1)
    print(f"[INFO] Top-{len(top_symbols)} by market cap loaded (first 10): {top_symbols[:10]}")

    # 2) fetch snapshots for those symbols
    df_all = fetch_snapshots_for_symbols(args.db, top_symbols, cols=['snapshot_time', 'symbol'] + args.metrics)
    if df_all.empty:
        print("[FATAL] No snapshots for the selected symbols.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(df_all)} snapshot rows for {len(df_all['symbol'].unique())} symbols.")

    # 3) compute first N hours movement on alt_rank
    movers_df = compute_first_n_hours_movement(df_all, hours=args.hours)
    if movers_df.empty:
        print("[WARN] Could not compute first-hours movements (missing alt_rank?). Exiting.")
        sys.exit(1)

    # 4) select top-K movers
    k = min(args.select_k, len(movers_df))
    selected_df = movers_df.head(k).copy()
    selected = selected_df['symbol'].tolist()
    print(f"[INFO] Selected top-{k} movers by abs AltRank movement in first {args.hours} hours:")
    print(selected)
    selected_df.to_csv(f"{args.output_prefix}_selected_coins.csv", index=False)
    print(f"[OK] Selected coins exported to {args.output_prefix}_selected_coins.csv")

    # 5) restrict df_all to selected symbols
    df_selected = df_all[df_all['symbol'].isin(selected)].copy().reset_index(drop=True)
    if df_selected.empty:
        print("[FATAL] No snapshot rows for selected movers. Exiting.")
        sys.exit(1)

    # 6) compute time-bucketed means across selected movers
    df_bucket_mean, df_with_buckets = time_bucket_and_average(df_selected, freq=args.freq, metrics=args.metrics)
    if not df_bucket_mean.empty:
        df_bucket_mean.to_csv(f"{args.output_prefix}_bucketed_means.csv", index=False)
        print(f"[OK] Bucketed means written to {args.output_prefix}_bucketed_means.csv")
    else:
        print("[WARN] No bucketed means computed (metrics missing?)")

    # 7) correlations: raw (coin-snapshot rows) and log1p transformed
    metrics_present = [c for c in args.metrics if c in df_selected.columns]
    if len(metrics_present) < 2:
        print("[FATAL] Not enough metric columns present to compute correlations.")
        sys.exit(1)

    corr_raw = safe_corr_matrix(df_selected, metrics_present)
    df_log = df_selected.copy()
    for c in metrics_present:
        if pd.api.types.is_numeric_dtype(df_log[c]) and c in ('social_volume_24h', 'interactions_24h', 'price'):
            df_log[c] = df_log[c].clip(lower=-0.999).fillna(0.0).astype(float)
            df_log[c] = np.log1p(df_log[c])
    corr_log = safe_corr_matrix(df_log, metrics_present)

    # 8) run chronological first-hit backtest across the selected movers
    strategy_summary_df, per_symbol_outcomes_df = run_first_hit_backtest(
        df_selected,
        symbol_col='symbol',
        time_col='snapshot_time',
        price_col='price',
        strategies=None,
        initial_investment=args.initial_investment
    )

    per_symbol_outcomes_df.to_csv(f"{args.output_prefix}_backtest_per_symbol.csv", index=False)
    strategy_summary_df.to_csv(f"{args.output_prefix}_backtest_summary.csv")
    print(f"[OK] Backtest per-symbol outcomes -> {args.output_prefix}_backtest_per_symbol.csv")
    print(f"[OK] Backtest summary -> {args.output_prefix}_backtest_summary.csv")

    # 9) Produce PDF with timeseries, heatmaps and backtest summary
    with PdfPages(args.output_pdf) as pdf:
        # title page
        title = f"Aggregate analysis for top-{len(top_symbols)} → top-{k} AltRank movers\n" \
                f"Window: first {args.hours} hours from coin's first snapshot\n" \
                f"Metrics: {', '.join(args.metrics)}\n" \
                f"Time-bucket freq: {args.freq}\n" \
                f"Generated: {datetime.utcnow().isoformat()}Z"
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, title, va='top', ha='left', wrap=True, fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # time-series of bucketed means
        if not df_bucket_mean.empty:
            fig, ax1 = plt.subplots(figsize=(14, 6))
            if 'alt_rank' in df_bucket_mean.columns:
                ax1.plot(df_bucket_mean['time_bucket'], df_bucket_mean['alt_rank'], label='mean(alt_rank)')
                ax1.set_ylabel('Mean AltRank (lower = better)')
                ax1.invert_yaxis()
            ax2 = ax1.twinx()
            if 'price' in df_bucket_mean.columns:
                ax2.plot(df_bucket_mean['time_bucket'], df_bucket_mean['price'], label='mean(price)', linestyle='--')
            if 'galaxy_score' in df_bucket_mean.columns:
                ax2.plot(df_bucket_mean['time_bucket'], df_bucket_mean['galaxy_score'], label='mean(galaxy_score)', linestyle=':')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax1.set_xlabel('Time bucket (' + args.freq + ')')
            plt.title(f"Mean metrics across selected movers (bucketed by {args.freq})")
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

            # social metrics page (symlog)
            fig, ax = plt.subplots(figsize=(14, 6))
            if 'social_volume_24h' in df_bucket_mean.columns:
                ax.plot(df_bucket_mean['time_bucket'], df_bucket_mean['social_volume_24h'], label='mean(social_volume_24h)')
            if 'interactions_24h' in df_bucket_mean.columns:
                ax.plot(df_bucket_mean['time_bucket'], df_bucket_mean['interactions_24h'], label='mean(interactions_24h)')
            ax.set_xlabel('Time bucket (' + args.freq + ')')
            ax.set_ylabel('Mean counts (social / interactions)')
            ax.set_yscale('symlog')
            ax.legend(loc='upper right')
            plt.title("Mean social metrics across selected movers (symlog)")
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # correlation heatmaps
        def plot_corr_matrix(corr_df, title_text):
            if corr_df.empty:
                fig, ax = plt.subplots(figsize=(8.5, 6))
                ax.text(0.5, 0.5, f"{title_text}\n(not available)", ha='center', va='center')
                ax.axis('off'); pdf.savefig(fig); plt.close(fig)
                return
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_df, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr_df.columns)))
            ax.set_yticks(np.arange(len(corr_df.columns)))
            ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_df.columns)
            plt.colorbar(im, ax=ax)
            for i in range(len(corr_df.columns)):
                for j in range(len(corr_df.columns)):
                    val = corr_df.iloc[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if abs(val) > 0.5 else 'black')
            ax.set_title(title_text)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        plot_corr_matrix(corr_raw, "Correlation matrix (raw, coin-snapshot rows)")
        plot_corr_matrix(corr_log, "Correlation matrix (log1p transformed for skewed metrics)")

        # backtest summary page
        fig = plt.figure(figsize=(11, 14))
        fig.suptitle('Fixed-Percentage Backtest Summary (selected movers)', fontsize=16, y=0.96)
        ax1 = fig.add_subplot(2, 1, 1)
        if not strategy_summary_df.empty:
            bars = ax1.bar(strategy_summary_df.index.astype(str), strategy_summary_df['total_pnl_usd'],
                           color=['green' if x >= 0 else 'red' for x in strategy_summary_df['total_pnl_usd']])
            ax1.axhline(0, color='black', linewidth=0.8)
            ax1.set_title(f'Total P&L (per trade ${args.initial_investment})', fontsize=12)
            ax1.set_ylabel('Total $ P&L')
            ax1.grid(axis='y', linestyle='--', linewidth=0.6)
            plt.xticks(rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, "No backtest summary available", ha='center', va='center')
            ax1.axis('off')

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.axis('off')
        if not strategy_summary_df.empty:
            table_df = strategy_summary_df.reset_index().round(2)
            tbl = ax2.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.05, 1.2)
        else:
            ax2.text(0.5, 0.5, "No strategy table", ha='center', va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig); plt.close(fig)

        # final page: list selected coins + their movement
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        txt = "Selected movers (top {}) by abs(AltRank movement in first {} hours):\n\n".format(k, args.hours)
        for i, r in selected_df.iterrows():
            txt += f"{i+1}. {r['symbol']} | movement: {r['abs_movement']:.2f} | samples_in_window: {r['samples_in_window']} | t0: {r['t0']}\n"
        ax.text(0.03, 0.98, txt, va='top', ha='left', wrap=True, fontsize=9)
        pdf.savefig(fig); plt.close(fig)

    print(f"[OK] PDF report saved to {args.output_pdf}")

if __name__ == "__main__":
    main()
